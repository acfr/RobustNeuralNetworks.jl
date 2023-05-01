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
    nu::Int, nh::Vector{Int}, ny::Int, γ::Number;
    nl::Function = Flux.relu, 
    initW::Function = Flux.glorot_normal,
    initb::Function = Flux.glorot_normal,
    rng::AbstractRNG = Random.GLOBAL_RNG
) where T

    # Direct parameterisation
    direct = DirectLBDNParams{T}(
        nu, nh, ny;
        initW=initW, initb=initb, rng=rng
    )

    return DenseLBDNParams{T}(nl, nu, nh, ny, T(γ), direct)

end

Flux.trainable(m::DenseLBDNParams) = Flux.trainable(m.direct)

# TODO: Add GPU compatibility
# Flux.cpu() ...
# Flux.gpu() ...

# TODO: Improve speed here
function cayley(X, Y)
    Z = X - X' + Y'*Y
    IZ = (I + Z)
    A = IZ \ (I - Z)
    B = -2Y / IZ
    return A, B
end

_get_A(AB_k) = AB_k[1]
_get_B(AB_k) = AB_k[2]'
_get_Ψ(d_k) = exp.(d_k)

# TODO: Improve speed (particularly around Cayley transform)
function direct_to_explicit(ps::DenseLBDNParams{T}) where T

    # Easy ones
    b = ps.direct.b
    Ψ = _get_Ψ.(ps.direct.d)

    # Cayley for A and B
    AB = cayley.(ps.direct.X, ps.direct.Y)
    A = _get_A.(AB)
    B = _get_B.(AB)

    return ExplicitLBDNParams{T}(A, B, Ψ, b)

end



