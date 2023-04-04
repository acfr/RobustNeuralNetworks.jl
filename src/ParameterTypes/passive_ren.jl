mutable struct PassiveRENParams{T} <: AbstractRENParams{T}
    nl                          # Sector-bounded nonlinearity
    nu::Int
    nx::Int
    nv::Int
    ny::Int
    direct::DirectParams{T}
    αbar::T
    ν::T
    # TODO: Add a field for incrementally strictly output passive model (ρ)
end

"""
    PassiveRENParams{T}(nu, nx, nv, ny; <keyword arguments>) where T

Construct direct parameterisation of a passive REN.

# Arguments
- `nu::Int`: Number of inputs.
- `nx::Int`: Number of states.
- `nv::Int`: Number of neurons.
- `ny::Int`: Number of outputs.
    
# Keyword arguments

- `ν::T=0`: Passivity parameter. Use ν>0 for incrementally strictly input passive model, and ν == 0 for incrementally passive model. 

- `nl=Flux.relu`: Static nonlinearity (eg: `Flux.relu` or `Flux.tanh`).

- `αbar::T=1`: Upper bound on the contraction rate with `ᾱ ∈ (0,1]`.

See [`DirectParams`](@ref) documentation for arguments `init`, `ϵ`, `bx_scale`, `bv_scale`, `polar_param`, `rng`.

See also [`GeneralRENParams`](@ref), [`ContractingRENParams`](@ref), [`LipschitzRENParams`](@ref).
"""
function PassiveRENParams{T}(
    nu::Int, nx::Int, nv::Int, ny::Int;
    ν::T = T(0),
    nl = Flux.relu, 
    αbar::T = T(1),
    init = :random,
    polar_param::Bool = true,
    bx_scale::T = T(0), 
    bv_scale::T = T(1), 
    ϵ::T = T(1e-12), 
    rng::AbstractRNG = Random.GLOBAL_RNG
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

    return PassiveRENParams{T}(nl, nu, nx, nv, ny, direct_ps, αbar, ν)

end

function passive_trainable(L::DirectParams)
    ps = [L.ρ, L.X, L.Y1, L.X3, L.Y3, L.Z3, L.B2, L.C2, L.D12, L.D21, L.bx, L.bv, L.by]
    !(L.polar_param) && popfirst!(ps)
    return filter(p -> length(p) !=0, ps)
end

Flux.trainable(m::PassiveRENParams) = passive_trainable(m.direct)

function Flux.gpu(m::PassiveRENParams{T}) where T
    # TODO: Test and complete this
    direct_ps = Flux.gpu(m.direct)
    return PassiveRENParams{T}(
        m.nl, m.nu, m.nx, m.nv, m.ny, direct_ps, m.αbar, m.ν
    )
end

function Flux.cpu(m::PassiveRENParams{T}) where T
    # TODO: Test and complete this
    direct_ps = Flux.cpu(m.direct)
    return PassiveRENParams{T}(
        m.nl, m.nu, m.nx, m.nv, m.ny, direct_ps, m.αbar, m.ν
    )
end

function direct_to_explicit(ps::PassiveRENParams{T}, return_h=false) where T

    # System sizes
    nu = ps.nu
    nx = ps.nx
    ny = ps.ny
    ν = ps.ν
        
    # Implicit parameters
    ϵ = ps.direct.ϵ
    ρ = ps.direct.ρ
    X = ps.direct.X

    X3 = ps.direct.X3
    Y3 = ps.direct.Y3

    # Implicit system and output matrices
    B2_imp = ps.direct.B2
    D12_imp = ps.direct.D12

    C2 = ps.direct.C2
    D21 = ps.direct.D21

    # Constructing D22 for incrementally passive and incrementally strictly input passive. 
    # See Eqns 31-33 of TAC paper 
    # Currently converts to Hermitian to avoid numerical conditioning issues
    M = X3'*X3 + Y3 - Y3' + ϵ*I

    D22 = ν*Matrix(I, ny,nu) + M
    D21_imp = D21 - D12_imp'

    𝑅 = -2ν * Matrix(I, nu, nu) + D22 + D22'

    Γ2 = [C2'; D21_imp'; B2_imp] * (𝑅 \ [C2 D21_imp B2_imp'])

    if ps.direct.polar_param 
        # See Eqns 29 of TAC paper 
        H = exp(ρ[1])*X'*X / norm(X)^2 + Γ2
    else
        H = X'*X + ϵ*I + Γ2
    end

    # Get explicit parameterisation
    !return_h && (return hmatrix_to_explicit(ps, H, D22))
    return H

end
