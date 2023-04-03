"""
$(TYPEDEF)

Parameter struct to build an acyclic REN with behavioural
constraints encoded in Q, S, R matrices
"""
mutable struct GeneralRENParams{T} <: AbstractRENParams{T}
    nl                          # Sector-bounded nonlinearity
    nu::Int
    nx::Int
    nv::Int
    ny::Int
    direct::DirectParams{T}
    αbar::T
    Q::Matrix{T}
    S::Matrix{T}
    R::Matrix{T}
end

"""
    GeneralRENParams(nu::Int, nx::Int, nv::Int, ny::Int, 
                     Q::Matrix{T}, S::Matrix{T}, R::Matrix{T}; ...) where T

Main constructor for `GeneralRENParams`. Main arguments are:

- `nu`: Number of inputs
- `nx`: Number of states
- `nv`: Number of neurons
- `ny`: Number of outputs
- `Q`, `S`, `R`: Matrices to encode incremental quadratic constraints
    
Takes the following keyword arguments:

- `nl` (default `Flux.relu`): Static nonlinearity to use

- `αbar` (default `1`):  `ᾱ ∈ (0,1]` is the upper bound on the contraction rate.

- See documentation for `DirectParams` constructor for arguments `init`, `ϵ`, 
`bx_scale`, `bv_scale`, `polar_param`, `rng`.
"""
function GeneralRENParams{T}(
    nu::Int, nx::Int, nv::Int, ny::Int,
    Q::Matrix{T}, S::Matrix{T}, R::Matrix{T};
    nl = Flux.relu, 
    αbar = T(1),
    init = :random,
    polar_param = true,
    bx_scale = T(0), 
    bv_scale = T(1), 
    ϵ = T(1e-12), 
    rng = Random.GLOBAL_RNG
) where T

    # Check conditions on Q
    if !isposdef(-Q)
        Q = Q - ϵ*I
        if ~isposdef(-Q)
            error("Q must be negative semi-definite for this construction.")
        end
    end

    # Check conditions on S and R
    if !ishermitian(R - S * (Q \ S'))
        Lr = R - S * (Q \ S')
        if maximum(abs.(Lr .- Lr')) < 1e-10
            @warn """Matrix R - S * (Q \\ S') is not Hermitian due to 
            numerical conditioning issues. Will convert to Hermitian matrix."""
        else
            error("Matrix R - S * (Q \\ S') must be Hermitian.")
        end
    end

    # Direct (implicit) params
    direct_ps = DirectParams{T}(
        nu, nx, nv, ny; 
        init=init, ϵ=ϵ, bx_scale=bx_scale, bv_scale=bv_scale, 
        polar_param=polar_param, D22_free=false, rng=rng
    )

    return GeneralRENParams{T}(nl, nu, nx, nv, ny, direct_ps, αbar, Q, S, R)

end

"""
    Flux.trainable(m::GeneralRENParams)

Define trainable parameters for `GeneralRENParams` type
"""
Flux.trainable(m::GeneralRENParams) = Flux.trainable(m.direct)

"""
    Flux.gpu(m::GeneralRENParams{T}) where T

Add GPU compatibility for `GeneralRENParams` type
"""
function Flux.gpu(m::GeneralRENParams{T}) where T
    direct_ps = Flux.gpu(m.direct)
    return GeneralRENParams{T}(
        m.nl, m.nu, m.nx, m.nv, m.ny, direct_ps, m.αbar, m.Q, m.S, m.R
    )
end

"""
    Flux.cpu(m::GeneralRENParams{T}) where T

Add CPU compatibility for `GeneralRENParams` type
"""
function Flux.cpu(m::GeneralRENParams{T}) where T
    direct_ps = Flux.cpu(m.direct)
    return GeneralRENParams{T}(
        m.nl, m.nu, m.nx, m.nv, m.ny, direct_ps, m.αbar, m.Q, m.S, m.R
    )
end

"""
    direct_to_explicit(ps::GeneralRENParams, return_h=false) where T

Convert direct REN parameterisation to explicit parameterisation
using behavioural constraints encoded in Q, S, R.

If `return_h = false` (default), function returns an object of type
`ExplicitParams{T}`. If `return_h = true`, returns the H matrix directly. 
Useful for debugging or model analysis.
"""
function direct_to_explicit(ps::GeneralRENParams{T}, return_h=false) where T

    # System sizes
    nu = ps.nu
    nx = ps.nx
    ny = ps.ny

    # Dissipation parameters
    Q = ps.Q
    S = ps.S
    R = ps.R

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
    # Currently converts to Hermitian to avoid numerical conditioning issues
    LQ = Matrix{T}(cholesky(-Q).U)
    R1 = Hermitian(R - S * (Q \ S'))
    LR = Matrix{T}(cholesky(R1).U) 
    
    M = X3'*X3 + Y3 - Y3' + Z3'*Z3 + ϵ*I
    if ny >= nu
        N = [(I - M) / (I + M); -2*Z3 / (I + M)]
    else
        N = [((I + M) \ (I - M)) (-2*(I + M) \ Z3')]
    end

    D22 = -(Q \ S') + (LQ \ N) * LR

    # Constructing H. See Eqn 28 of TAC paper
    C2_imp = (D22'*Q + S)*C2
    D21_imp = (D22'*Q + S)*D21 - D12_imp'

    𝑅 = R + S*D22 + D22'*S' + D22'*Q*D22

    Γ1 = [C2'; D21'; zeros(nx, ny)] * Q * [C2 D21 zeros(ny, nx)]
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
