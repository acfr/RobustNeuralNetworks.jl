mutable struct DenseLBDNParams{T} <: AbstractLBDNParams{T}
    nl::Function                    # Sector-bounded nonlinearity
    nu::Int
    nh::Vector{Int}
    ny::Int
    γ::T
    direct::DirectLBDNParams{T}
end

# Constructor for dense LBDN params
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

Flux.trainable(m::DenseLBDNParams) = Flux.trainable(m.direct)

# Scaling layer
mutable struct Scale{T}
    scale::T
end
(m::Scale)(u::AbstractVecOrMat) = m.scale .* u

# Output layer
mutable struct DenseLBDNOutput{T}
    B::AbstractMatrix{T}
    b::AbstractVector{T}
end
(m::DenseLBDNOutput)(u::AbstractVecOrMat) = m.B * u .+ m.b

# Dense LBDN layer (explicit model)
mutable struct DenseLBDNLayer{T}
    nl ::Function
    A_T::AbstractMatrix{T}              # A^T in the paper
    B  ::AbstractMatrix{T}
    Ψd ::AbstractVector{T}              # Diagonal of matrix Ψ from the paper
    b  ::AbstractVector{T}
end

# Default constructor
function DenseLBDNLayer{T}() where T
    DenseLBDNLayer{T}(Flux.relu, zeros(T,0,0), zeros(T,0,0), zeros(T,0), zeros(T,0))
end

# Call the dense LBDN layer
function (m::DenseLBDNLayer)(u::AbstractVecOrMat{T}) where T
    sqrt2 = T(√2)
    return sqrt2 * m.A_T .* m.Ψd * m.nl.(sqrt2 ./ m.Ψd .* (m.B * u) .+ m.b)
end

function direct_to_explicit(ps::DenseLBDNParams{T}) where T

    # Direct parameterisation
    γ  = ps.γ
    σ  = ps.nl
    nh = ps.nh
    ny = ps.ny
    L  = length(nh)
    L1 = L + 1

    XY = ps.direct.XY
    α  = ps.direct.α
    d  = ps.direct.d
    b  = ps.direct.b

    # Build up a list of explicit sandwich layers
    # Use Zygote buffer to avoid issues with array mutation
    buf = Buffer([DenseLBDNLayer{T}()], L)
    for k in 1:L
        bk       = b[k]
        Ψd       = exp.(d[k])
        A_T, B_T = norm_cayley(XY[k], α[k], nh[k])
        buf[k]   = DenseLBDNLayer{T}(σ, A_T, B_T', Ψd, bk)
    end
    dense_layers = copy(buf)

    # Define the scaling and output layers
    scale_in   = Scale{T}(sqrt(γ))
    scale_out  = Scale{T}(sqrt(γ))

    AB         = norm_cayley(XY[L1], α[L1], ny)
    linear_out = DenseLBDNOutput{T}(AB[2]', b[L1])

    # Build a chain for LBDN model
    return Flux.Chain(
        scale_in, 
        dense_layers..., 
        scale_out, 
        linear_out
    )

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

# TODO: Add GPU compatibility
# Flux.cpu() ...
# Flux.gpu() ...
