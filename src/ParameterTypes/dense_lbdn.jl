# This file is a part of RobustNeuralNetworks.jl. License is MIT: https://github.com/acfr/RobustNeuralNetworks.jl/blob/main/LICENSE 

mutable struct DenseLBDNParams{T, L} <: AbstractLBDNParams{T, L}
    nl::Function                    # Sector-bounded nonlinearity
    nu::Int
    nh::NTuple{L, Int}
    ny::Int
    direct::DirectLBDNParams{T}
end

"""
    DenseLBDNParams{T}(nu, nh, ny, γ; <keyword arguments>) where T

Construct direct parameterisation of a dense (fully-connected) LBDN.

This is the equivalent of a multi-layer perceptron (eg: `Flux.Dense`) with a guaranteed Lipschitz bound of `γ`. Note that the Lipschitz bound can made a learnable parameter.

# Arguments
- `nu::Int`: Number of inputs.
- `nh::Union{Vector{Int}, NTuple{N, Int}}`: Number of hidden units for each layer. Eg: `nh = [5,10]` for 2 hidden layers with 5 and 10 nodes (respectively).
- `ny::Int`: Number of outputs.
- `γ::Real=T(1)`: Lipschitz upper bound, must be positive.

# Keyword arguments:

- `nl::Function=relu`: Sector-bounded static nonlinearity.
- `learn_γ::Bool=false:` Whether to make the Lipschitz bound γ a learnable parameter.

See [`DirectLBDNParams`](@ref) for documentation of keyword arguments `initW`, `initb`, `rng`.

"""
function DenseLBDNParams{T}(
    nu::Int, nh::Union{Vector{Int}, NTuple{N, Int}}, 
    ny::Int, γ::Real = T(1);
    nl::Function     = relu, 
    initW::Function  = glorot_normal,
    initb::Function  = glorot_normal,
    learn_γ::Bool    = false,
    rng::AbstractRNG = Random.GLOBAL_RNG
) where {T, N}
    nh = Tuple(nh)
    direct = DirectLBDNParams{T}(nu, nh, ny, γ; initW, initb, learn_γ, rng)
    return DenseLBDNParams{T, length(nh)}(nl, nu, nh, ny, direct)
end

@functor DenseLBDNParams
trainable(m::DenseLBDNParams) =  (direct = m.direct, )

function direct_to_explicit(ps::DenseLBDNParams{T, L}) where {T, L}

    # Direct parameterisation
    nh = ps.nh
    ny = ps.ny
    L1 = L + 1

    XY = ps.direct.XY
    α  = ps.direct.α
    d  = ps.direct.d
    b  = ps.direct.b
    log_γ = ps.direct.log_γ

    # Build explicit model
    Ψd     = get_Ψ(d)
    A_T, B = get_AB(XY, α, vcat(nh..., ny))
    sqrtγ  = sqrt.(exp.(log_γ))

    # Faster to backpropagate with tuples than vectors
    return ExplicitLBDNParams{T,L1,L}(tuple(A_T...), tuple(B...), tuple(Ψd...), b, sqrtγ)

end

function norm_cayley(XY, α, n)

    # Normalise XY with polar param and extract
    XY = (α ./ norm(XY)) .* XY
    X  = XY[1:n, :]
    Y  = XY[(n+1):end, :]

    # Cayley transform
    Z = (X - X') + (Y'*Y)
    Iz = _get_I(Z) # Prevents scalar indexing on backwards pass of A / (I + Z) on GPU
    A_T = (Iz + Z) \ (Iz - Z)
    B_T = -2Y / (Iz + Z)

    return A_T, B_T
    
end

function get_Ψ(d::NTuple{N, T}) where {N, T}

    # Use Zygote buffer to avoid problems with array mutation
    buf = Buffer([zero(d[1])], N)
    for k in 1:N
        buf[k] = exp.(d[k])
    end
    return copy(buf)

end

function get_AB(
    XY::NTuple{N, T1}, 
    α ::NTuple{N, T2},
    n ::Vector{Int}
) where {N, T1, T2}

    # Use Zygote buffer to avoid problems with array mutation
    buf_A = Buffer([zero(XY[1])], N)
    buf_B = Buffer([zero(XY[1])], N)
    for k in 1:N
        AB_k = norm_cayley(XY[k], α[k], n[k])
        buf_A[k] = AB_k[1]
        buf_B[k] = AB_k[2]'
    end
    return copy(buf_A), copy(buf_B)

end
