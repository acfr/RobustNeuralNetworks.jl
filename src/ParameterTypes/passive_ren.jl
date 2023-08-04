# This file is a part of RobustNeuralNetworks.jl. License is MIT: https://github.com/acfr/RobustNeuralNetworks.jl/blob/main/LICENSE 

mutable struct PassiveRENParams{T} <: AbstractRENParams{T}
    nl::Function                # Sector-bounded nonlinearity
    nu::Int
    nx::Int
    nv::Int
    ny::Int
    direct::DirectRENParams{T}
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

- `nl::Function=relu`: Sector-bounded static nonlinearity.

- `αbar::T=1`: Upper bound on the contraction rate with `ᾱ ∈ (0,1]`.

See [`DirectRENParams`](@ref) for documentation of keyword arguments `init`, `ϵ`, `bx_scale`, `bv_scale`, `polar_param`, `rng`.

See also [`GeneralRENParams`](@ref), [`ContractingRENParams`](@ref), [`LipschitzRENParams`](@ref).
"""
function PassiveRENParams{T}(
    nu::Int, nx::Int, nv::Int, ny::Int;
    ν::T              = T(0),
    nl::Function      = relu, 
    αbar::T           = T(1),
    init              = :random,
    polar_param::Bool = true,
    bx_scale::T       = T(0), 
    bv_scale::T       = T(1), 
    ϵ::T              = T(1e-12), 
    rng::AbstractRNG  = Random.GLOBAL_RNG
) where T

    # Check input output pair dimensions
    if nu != ny
        error("Input and output must have the same dimension for passiveREN")
    end

    # Direct (implicit) params
    direct_ps = DirectRENParams{T}(
        nu, nx, nv, ny; 
        init, ϵ, bx_scale, bv_scale, polar_param, 
        D22_free=false, rng,
    )

    return PassiveRENParams{T}(nl, nu, nx, nv, ny, direct_ps, αbar, ν)

end

@functor PassiveRENParams (direct, )

function direct_to_explicit(ps::PassiveRENParams{T}, return_h=false) where T

    # System sizes
    nu = ps.nu
    ny = ps.ny
    ν = ps.ν
        
    # Implicit parameters
    ϵ = ps.direct.ϵ
    ρ = ps.direct.ρ[1]
    X = ps.direct.X
    polar_param = ps.direct.polar_param

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
    M = _M_pass(X3, Y3, ϵ)

    D22 = ν*Matrix(I, ny,nu) + M
    D21_imp = D21 - D12_imp'

    𝑅  = _R_pass(nu, D22, ν)
    Γ2 = _Γ2_pass(C2, D21_imp, B2_imp, 𝑅)

    H = x_to_h(X, ϵ, polar_param, ρ) + Γ2

    # Get explicit parameterisation
    !return_h && (return hmatrix_to_explicit(ps, H, D22))
    return H

end

_M_pass(X3, Y3, ϵ) = X3'*X3 + Y3 - Y3' + ϵ*I

_R_pass(nu, D22, ν) = -2ν * Matrix(I, nu, nu) + D22 + D22'

function _Γ2_pass(C2, D21_imp, B2_imp, 𝑅)
    [C2'; D21_imp'; B2_imp] * (𝑅 \ [C2 D21_imp B2_imp'])
end