"""
$(TYPEDEF)

Parameter struct to build an acyclic REN with a guaranteed
Lipschitz bound of γ ∈ ℝ
"""
mutable struct LipschitzRENParams{T} <: AbstractRENParams{T}
    nl                          # Sector-bounded nonlinearity
    nu::Int
    nx::Int
    nv::Int
    ny::Int
    direct::DirectParams{T}
    αbar::T
    γ::T
end

"""
    LipschitzRENParams(nu, nx, nv, ny; ...)

Main constructor for `LipschitzRENParams`.
ᾱ ∈ (0,1] is the upper bound on contraction rate.
"""
function LipschitzRENParams{T}(
    nu::Int, nx::Int, nv::Int, ny::Int, γ::Number;
    init = :random,
    nl = Flux.relu, 
    ϵ = T(1e-12), 
    αbar = T(1),
    bx_scale = T(0), 
    bv_scale = T(1), 
    polar_param = true,
    D22_zero = false,
    rng = Random.GLOBAL_RNG
) where T

    # If D22 fixed at 0, it should not be constructed from other
    # direct params (so set D22_free = true)
    D22_free = D22_zero ? true : false

    # Direct (implicit) params
    direct_ps = DirectParams{T}(
        nu, nx, nv, ny; 
        init=init, ϵ=ϵ, bx_scale=bx_scale, bv_scale=bv_scale, 
        polar_param=polar_param, D22_free=D22_free, D22_zero=D22_zero,
        rng=rng
    )

    return LipschitzRENParams{T}(nl, nu, nx, nv, ny, direct_ps, αbar, T(γ))

end

"""
    Flux.trainable(m::LipschitzRENParams)

Define trainable parameters for `LipschitzRENParams` type
"""
Flux.trainable(m::LipschitzRENParams) = Flux.trainable(m.direct)

"""
    Flux.gpu(m::LipschitzRENParams{T}) where T

Add GPU compatibility for `LipschitzRENParams` type
"""
function Flux.gpu(m::LipschitzRENParams{T}) where T
    direct_ps = Flux.gpu(m.direct)
    return LipschitzRENParams{T}(
        m.nl, m.nu, m.nx, m.nv, m.ny, direct_ps, m.αbar, m.γ
    )
end

"""
    Flux.cpu(m::LipschitzRENParams{T}) where T

Add CPU compatibility for `LipschitzRENParams` type
"""
function Flux.cpu(m::LipschitzRENParams{T}) where T
    direct_ps = Flux.cpu(m.direct)
    return LipschitzRENParams{T}(
        m.nl, m.nu, m.nx, m.nv, m.ny, direct_ps, m.αbar, m.γ
    )
end

"""
    direct_to_explicit(ps::LipschitzRENParams, return_h=false) where T

Convert direct REN parameterisation to explicit parameterisation
using Lipschitz bounded behavioural constraint.

If `return_h = false` (default), function returns an object of type
`ExplicitParams{T}`. If `return_h = true`, returns the H matrix directly. 
Useful for debugging or model analysis.
"""
function direct_to_explicit(ps::LipschitzRENParams{T}, return_h=false) where T

    # System sizes
    nu = ps.nu
    nx = ps.nx
    ny = ps.ny

    # Dissipation parameters
    γ = ps.γ

    # Implicit parameters
    ϵ = ps.direct.ϵ
    ρ = ps.direct.ρ
    X = ps.direct.X

    X3 = ps.direct.X3
    Y3 = ps.direct.Y3
    Z3 = ps.direct.Z3

    # Implicit system and output matrices
    B2_imp = ps.direct.B2
    D12_imp = ps.direct.D12

    C2 = ps.direct.C2
    D21 = ps.direct.D21

    # Constructing D22. See Eqns 31-33 of TAC paper
    if ps.direct.D22_zero
        D22 = ps.direct.D22
    else
        M = X3'*X3 + Y3 - Y3' + Z3'*Z3 + ϵ*I
        N = (ny >= nu) ? [(I - M) / (I + M); -2*Z3 / (I + M)] :
                        [((I + M) \ (I - M)) (-2*(I + M) \ Z3')]
        D22 = γ*N
    end

    # Constructing H. See Eqn 28 of TAC paper
    C2_imp = -(D22')*C2 / γ
    D21_imp = -(D22')*D21 / γ - D12_imp'

    𝑅 = -D22'*D22 / γ + (γ * I)

    Γ1 = [C2'; D21'; zeros(nx, ny)] * [C2 D21 zeros(ny, nx)] * (-1/γ)
    Γ2 = [C2_imp'; D21_imp'; B2_imp] * (𝑅 \ [C2_imp D21_imp B2_imp'])

    if ps.direct.polar_param 
        H = exp(ρ[1])*(X'*X + ϵ*I) / norm(X)^2 + Γ2 - Γ1
    else
        H = X'*X + ϵ*I + Γ2 - Γ1
    end

    # Get explicit parameterisation
    !return_h && (return hmatrix_to_explicit(ps, H, D22))
    return H

end
