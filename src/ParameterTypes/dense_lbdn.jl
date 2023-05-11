mutable struct DenseLBDNParams{T} <: AbstractLBDNParams{T}
    nl::Function                    # Sector-bounded nonlinearity
    nu::Int
    nh::Vector{Int}
    ny::Int
    γ::T
    direct::DirectLBDNParams{T}
end

"""
    DenseLBDNParams{T}(nu, nh, ny, γ; <keyword arguments>) where T

Construct direct parameterisation of a dense (fully-connected) LBDN.

This is the equivalent of a multi-layer perceptron (eg: `Flux.Dense`) with a guaranteed Lipschitz bound of `γ`.

# Arguments
- `nu::Int`: Number of inputs.
- `nh::Vector{Int}`: Number of hidden units for each layer. Eg: `nh = [5,10]` for 2 hidden layers with 5 and 10 nodes (respectively).
- `ny::Int`: Number of outputs.
- `γ::Number=T(1)`: Lipschitz upper bound.

# Keyword arguments:

- `nl::Function=Flux.relu`: Sector-bounded static nonlinearity.

See [`DirectLBDNParams`](@ref) for documentation of keyword arguments `initW`, `initb`, `rng`.

"""
function DenseLBDNParams{T}(
    nu::Int, nh::Vector{Int}, ny::Int, γ::Number = T(1);
    nl::Function = Flux.relu, 
    initW::Function = Flux.glorot_normal,
    initb::Function = Flux.glorot_normal,
    rng::AbstractRNG = Random.GLOBAL_RNG
) where T

    direct = DirectLBDNParams{T}(nu, nh, ny; initW=initW, initb=initb, rng=rng)
    return DenseLBDNParams{T}(nl, nu, nh, ny, T(γ), direct)

end

Flux.@functor DenseLBDNParams (direct, )

function direct_to_explicit(ps::DenseLBDNParams{T}) where T

    # Direct parameterisation
    nh = ps.nh
    ny = ps.ny
    L  = length(nh)
    L1 = L + 1

    XY = ps.direct.XY
    α  = ps.direct.α
    d  = ps.direct.d
    b  = ps.direct.b

    # Build explicit model
    Ψd     = get_Ψ(d)
    A_T, B = get_AB(XY, α, vcat(nh, ny))

    # Faster to backpropagate with tuples than vectors
    return ExplicitLBDNParams{T,L1,L}(tuple(A_T...), tuple(B...), tuple(Ψd...), b)

end

function norm_cayley(XY, α, n)

    # Normalise XY with polar param and extract
    XY = (α[1] / norm(XY)) * XY
    X  = XY[1:n, :]
    Y  = XY[(n+1):end, :]

    # Cayley transform
    Z = (X - X') + (Y'*Y)
    IZ = (I + Z)
    A_T = IZ \ (I - Z)
    B_T = -2Y / IZ

    return A_T, B_T

end

function get_Ψ(d::NTuple{N, AbstractVector{T}}) where {T, N}

    # Use Zygote buffer to avoid problems with array mutation
    buf = Buffer([zeros(T,0)], N)
    for k in 1:N
        buf[k] = exp.(d[k])
    end
    return copy(buf)

end

function get_AB(
    XY::NTuple{N, AbstractMatrix{T}}, 
    α ::NTuple{N, AbstractVector{T}},
    n ::Vector{Int}
) where {T, N}

    # Use Zygote buffer to avoid problems with array mutation
    buf_A = Buffer([zeros(T,0,0)], N)
    buf_B = Buffer([zeros(T,0,0)], N)
    for k in 1:N
        AB_k = norm_cayley(XY[k], α[k], n[k])
        buf_A[k] = AB_k[1]
        buf_B[k] = AB_k[2]'
    end
    return copy(buf_A), copy(buf_B)
    
end

# TODO: Add GPU compatibility
# Flux.cpu() ...
# Flux.gpu() ...