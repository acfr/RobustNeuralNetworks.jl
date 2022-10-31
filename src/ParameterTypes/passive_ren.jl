"""
$(TYPEDEF)

Parameter struct to build a passive REN where Q = 0, R  = 0, S = I
"""
mutable struct PassiveRENParams{T} <: AbstractRENParams{T}
    nl                          # Sector-bounded nonlinearity
    nu::Int
    nx::Int
    nv::Int
    ny::Int
    direct::DirectParams{T}
    output::OutputLayer{T}
    αbar::T
    ν::T
    # TODO: Add a filed for output passivity (ρ)
end

"""
    PassiveRENParams(nu, nx, nv, ny; ...)

Main constructor for `PassiveRENParams`.
ᾱ ∈ (0,1] is the upper bound on contraction rate.
ν>0 is to define incrementally input passive; v == 0 for incrementally passive model. 
"""
function PassiveRENParams{T}(
    nu::Int, nx::Int, nv::Int, ny::Int;
    init = :random,
    nl = Flux.relu, 
    ν = 0,
    ϵ = T(1e-6), 
    αbar = T(1),
    bx_scale = T(0), 
    bv_scale = T(1), 
    polar_param = true,
    rng = Random.GLOBAL_RNG
) where T

    # Check input output pair dimensions
    if nu != ny
        error("Input and output must have the same dimension for passiveREN")
    end

    # Direct (implicit) params
    direct_ps = DirectParams{T}(
        nu, nx, nv, ny; 
        init=init, ϵ=ϵ, bx_scale=bx_scale, bv_scale=bv_scale, 
        polar_param=polar_param, D22_free=false, rng=rng
    )

    # Output layer
    output_ps = OutputLayer{T}(nu, nx, nv, ny; D22_trainable=false, rng=rng)

    return PassiveRENParams{T}(nl, nu, nx, nv, ny, direct_ps, output_ps, αbar,ν)

end

"""
    Flux.trainable(m::PassiveRENParams)

Define trainable parameters for `PassiveRENParams` type
"""
Flux.trainable(m::PassiveRENParams) = [
    Flux.trainable(m.direct)..., Flux.trainable(m.output)...
]

"""
    Flux.gpu(m::PassiveRENParams{T}) where T

Add GPU compatibility for `PassiveRENParams` type
"""
function Flux.gpu(m::PassiveRENParams{T}) where T
    direct_ps = Flux.gpu(m.direct)
    output_ps = Flux.gpu(m.output)
    return PassiveRENParams{T}(
        m.nl, m.nu, m.nx, m.nv, m.ny, direct_ps, output_ps, m.αbar, m.Q, m.S, m.R
    )
end

"""
    Flux.cpu(m::PassiveRENParams{T}) where T

Add CPU compatibility for `PassiveRENParams` type
"""
function Flux.cpu(m::PassiveRENParams{T}) where T
    direct_ps = Flux.cpu(m.direct)
    output_ps = Flux.cpu(m.output)
    return PassiveRENParams{T}(
        m.nl, m.nu, m.nx, m.nv, m.ny, direct_ps, output_ps, m.αbar, m.Q, m.S, m.R
    )
end

"""
    direct_to_explicit(ps::PassiveRENParams)

Convert direct REN parameterisation to explicit parameterisation
using behavioural constraints encoded in Q, S, R
"""
function direct_to_explicit(ps::PassiveRENParams{T}) where T

    # System sizes
    nu = ps.nu
    nx = ps.nx
    ny = ps.ny
    ν = ps.ν
    
    # Dissipation IQC conditions
    Q = zeros(ny, ny)
    S = Matrix(I, nu, ny)
    R = -2ν * Matrix(I, nu, nu)
    
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

    C2 = ps.output.C2
    D21 = ps.output.D21

    # Constructing D22 for incrementally passive and incrementally strictly input passive. 
    # See Eqns 31-33 of TAC paper 
    # Currently converts to Hermitian to avoid numerical conditioning issues
    M = X3'*X3 + Y3 - Y3' + Z3'*Z3 + ϵ*I
    N = [(I - M) / (I + M); -2*Z3 / (I + M)]

    D22 = ν*Matrix(I, ny,nu) + M

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

    # Get explicit parameterisation
    return hmatrix_to_explicit(ps, H, D22)

end
