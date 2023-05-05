mutable struct DenseLBDNParams{T} <: AbstractLBDNParams{T}
    nl::Function                    # Sector-bounded nonlinearity
    nu::Int
    nh::Vector{Int}
    ny::Int
    γ::T
    direct::DirectLBDNParams{T}
end

# Constructor
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

function direct_to_explicit(ps::DenseLBDNParams{T}) where T

    # Easy ones
    L = length(ps.nh)
    b = _get_b.(ps.direct.b)
    Ψd = _get_Ψ.(ps.direct.d)

    # Normalise weights with polar param 
    # XY = ps.direct.α .* ps.direct.XY ./ norm.(ps.direct.XY)

    # Cayley for A and B
    AB = cayley.(ps.direct.XY, ps.direct.α, (ps.nh..., ps.ny))
    B = _get_B.(AB)
    A_T = _get_A.(AB)

    return ExplicitLBDNParams{T,L+1,L}(A_T, B, Ψd, b)

end

# TODO: Improve speed
function cayley(XY, α, n)

    XY = α[1] * XY ./ norm(XY)

    X = XY[1:n, :]
    Y = XY[(n+1):end, :]

    Z = X - X' + Y'*Y
    IZ = (I + Z)
    A_T = IZ \ (I - Z)
    B_T = -2Y / IZ

    return A_T, B_T

end

_get_b(b_k) = b_k
_get_Ψ(d_k) = exp.(d_k)
_get_A(AB_k) = AB_k[1]
_get_B(AB_k) = AB_k[2]'


# TODO: Add GPU compatibility
# Flux.cpu() ...
# Flux.gpu() ...