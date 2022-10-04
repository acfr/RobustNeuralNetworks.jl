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
    output::OutputLayer{T}
    αbar::T
    Q::Matrix{T}
    S::Matrix{T}
    R::Matrix{T}
end

"""
    GeneralRENParams(nu, nx, nv, ny; ...)

Main constructor for `GeneralRENParams`.
ᾱ ∈ (0,1] is the upper bound on contraction rate.
"""
function GeneralRENParams{T}(
    nu::Int, nx::Int, nv::Int, ny::Int,
    Q = nothing, S = nothing, R = nothing;
    init = :random,
    nl = Flux.relu, 
    ϵ = T(1e-6), 
    αbar = T(1),
    bx_scale = T(0), 
    bv_scale = T(1), 
    polar_param = true,
    rng = Random.GLOBAL_RNG
) where T

    # IQC params
    (Q === nothing) && (Q = Matrix{T}(-I, ny, ny))
    (S === nothing) && (S = zeros(T, nu, ny))
    (R === nothing) && (R = Matrix{T}(I, nu, nu))

    # Check conditions on Q
    if !isposdef(-Q)
        Q = Q - ϵ*I
        if ~isposdef(-Q)
            error("Q must be negative semi-definite for this construction.")
        end
    end

    # Direct (implicit) params
    direct_ps = DirectParams{T}(
        nu, nx, nv, ny; 
        init=init, ϵ=ϵ, bx_scale=bx_scale, bv_scale=bv_scale, 
        polar_param=polar_param, D22_free=false, rng=rng
    )

    # Output layer
    output_ps = OutputLayer{T}(nu, nx, nv, ny; D22_trainable=false, rng=rng)

    return GeneralRENParams{T}(nl, nu, nx, nv, ny, direct_ps, output_ps, αbar, Q, S, R)

end

"""
    Flux.trainable(m::GeneralRENParams)

Define trainable parameters for `GeneralRENParams` type
"""
Flux.trainable(m::GeneralRENParams) = [
    Flux.trainable(m.direct)..., Flux.trainable(m.output)...
]

"""
    Flux.gpu(m::GeneralRENParams{T}) where T

Add GPU compatibility for `GeneralRENParams` type
"""
function Flux.gpu(m::GeneralRENParams{T}) where T
    direct_ps = Flux.gpu(m.direct)
    output_ps = Flux.gpo(m.output)
    return GeneralRENParams{T}(
        m.nl, m.nu, m.nx, m.nv, m.ny, direct_ps, output_ps, m.αbar, m.Q, m.S, m.R
    )
end

"""
    Flux.cpu(m::GeneralRENParams{T}) where T

Add CPU compatibility for `GeneralRENParams` type
"""
function Flux.cpu(m::GeneralRENParams{T}) where T
    direct_ps = Flux.cpu(m.direct)
    output_ps = Flux.cpo(m.output)
    return GeneralRENParams{T}(
        m.nl, m.nu, m.nx, m.nv, m.ny, direct_ps, output_ps, m.αbar, m.Q, m.S, m.R
    )
end

"""
    direct_to_explicit(ps::GeneralRENParams)

Convert direct REN parameterisation to explicit parameterisation
using behavioural constraints encoded in Q, S, R
"""
function direct_to_explicit(ps::GeneralRENParams{T}) where T

    # System sizes
    nu = ps.nu
    nx = ps.nx
    nv = ps.nv
    ny = ps.ny

    # Dissipation parameters
    Q = ps.Q
    S = ps.S
    R = ps.R

    # Implicit parameters
    ᾱ = ps.αbar
    ϵ = ps.direct.ϵ
    ρ = ps.direct.ρ
    X = ps.direct.X

    Y1 = ps.direct.Y1

    X3 = ps.direct.X3
    Y3 = ps.direct.Y3
    Z3 = ps.direct.Z3

    # Implicit system and output matrices
    B2_imp = ps.direct.B2
    D12_imp = ps.direct.D12

    C2 = ps.output.C2
    D21 = ps.output.D21

    # Constructing D22. See Eqns 31-33 of TAC paper
    LQ = Matrix{T}(cholesky(-Q).U)
    LR = Matrix{T}(cholesky(R - S * (Q \ S')).U)

    M = X3'*X3 + Y3 - Y3' + Z3'*Z3 + ϵ*I
    if ny >= nu
        N = [(I - M) / (I + M); -2*Z3 / (I + M)]
    else
        N = [((I + M) \ (I - M)) (-2*(I + M) \ Z3')]
    end

    D22 = Q \ S' + LQ \ N * LR

    # Constructing H. See Eqn 28 of TAC paper
    C2_imp = (D22'*Q + S)*C2
    D21_imp = (D22'*Q + S)*D21 - D12_imp'

    𝑅 = R + S*D22 + D22'*S' + D22'*Q*D22

    Γ1 = [C2'; D21'; zeros(nx, ny)] * Q * [C2 D21 zeros(ny, nx)]
    Γ2 = [C2_imp'; D21_imp'; B2_imp] * (𝑅 \ [C2_imp D21_imp B2_imp'])

    if ps.direct.polar_param 
        H = exp(ρ[1])*X'*X / norm(X)^2 + Γ2 - Γ1
    else
        H = X'*X + ϵ*I + Γ2 - Γ1
    end

    # Extract sections of H matrix 
    # Note: using @view slightly faster, but not supported by CUDA
    H11 = H[1:nx, 1:nx]
    H22 = H[nx + 1:nx + nv, nx + 1:nx + nv]
    H33 = H[nx + nv + 1:2nx + nv, nx + nv + 1:2nx + nv]
    H21 = H[nx + 1:nx + nv, 1:nx]
    H31 = H[nx + nv + 1:2nx + nv, 1:nx]
    H32 = H[nx + nv + 1:2nx + nv, nx + 1:nx + nv]

    # Construct implicit model parameters
    P_imp = H33
    F = H31
    E = (H11 + P_imp/ᾱ^2 + Y1 - Y1')/2

    # Equilibrium network parameters
    B1_imp = H32
    C1_imp = -H21
    Λ_inv = (1 ./ diag(H22)) * 2
    D11_imp = -tril(H22, -1)

    # Construct the explicit model
    A = E \ F
    B1 = E \ B1_imp
    B2 = E \ ps.direct.B2
    
    C1 = Λ_inv .* C1_imp
    D11 = Λ_inv .* D11_imp
    D12 = Λ_inv .* D12_imp

    bx = ps.direct.bx
    bv = ps.direct.bv
    by = ps.output.by
    
    return ExplicitParams{T}(A, B1, B2, C1, C2, D11, D12, D21, D22, bx, bv, by)

end
