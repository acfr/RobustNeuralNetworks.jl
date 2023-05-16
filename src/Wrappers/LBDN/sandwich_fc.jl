mutable struct SandwichFC{F, T, D, B}
    σ ::F
    XY::AbstractMatrix{T}       # [X; Y] in the paper
    α ::AbstractVector{T}       # Polar parameterisation
    d ::D                       # d in paper
    b ::B                       # Bias vector
end

# Sandwich layer struct, meant to reflect Flux.Dense
function SandwichFC(
    (in, out)::Pair{<:Integer, <:Integer},
    σ::F                = Flux.identity;
    init::Function      = Flux.glorot_normal,
    bias::Bool          = true,
    output_layer::Bool  = false, 
    T::DataType         = Float32,
    rng::AbstractRNG    = Random.GLOBAL_RNG,
) where F

    # Matrices and polar param always required
    XY = init(rng, out + in, out)
    α  = [norm(XY)]

    # Only need d/bias if specified (AD knows `nothing` is not differentiable)
    d  = output_layer ? nothing : T.(init(rng, out))
    b  = bias ? T.(init(rng, out)) : nothing

    return SandwichFC{F, T, typeof(d), typeof(b)}(σ, XY, α, d, b)
end

Flux.@functor SandwichFC

# Call the sandwich layer
function (m::SandwichFC)(x::AbstractVecOrMat{T}) where T

    # Extract params
    XY = m.XY
    α = m.α
    d = m.d
    b = m.d
    σ = m.σ
    n = size(XY,2)

    bias = (b !== nothing)
    output = (d === nothing)

    # Get explicit model parameters
    A_T, B_T = norm_cayley(XY, α, n)
    B = B_T'
    
    # Just output layer?
    output && (return bias ? B*x .+ b : B*x)

    # Regular sandwich layer
    Ψd = exp.(d)
    sqrt2 = T(sqrt(2))
    
    x = sqrt2 * (B ./ Ψd) * x
    bias && (x = x .+ b)
    x = sqrt2 * (A_T .* Ψd') * σ.(x)

    return x

end