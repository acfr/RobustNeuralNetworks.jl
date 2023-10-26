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
    ρ::T
end

"""
    PassiveRENParams{T}(nu, nx, nv, ny, ν, ρ; <keyword arguments>) where T

Construct direct parameterisation of a passive REN.

# Arguments
- `nu::Int`: Number of inputs.
- `nx::Int`: Number of states.
- `nv::Int`: Number of neurons.
- `ny::Int`: Number of outputs.
- `ν::Number=0`: Passivity index. Use ν>0 for incrementally strictly input passive model. Setting both ν == 0 and ρ == 0 for incrementally passive model.
- `ρ::Number=0`: Passivity index. Use ρ>0 for incrementally strictly output passive model. Note: setting both ν and ρ>0 is not currently supported.

# Keyword arguments

- `nl::Function=relu`: Sector-bounded static nonlinearity.

- `αbar::T=1`: Upper bound on the contraction rate with `ᾱ ∈ (0,1]`.

See [`DirectRENParams`](@ref) for documentation of keyword arguments `init`, `ϵ`, `bx_scale`, `bv_scale`, `polar_param`, `rng`.

See also [`GeneralRENParams`](@ref), [`ContractingRENParams`](@ref), [`LipschitzRENParams`](@ref).
"""
function PassiveRENParams{T}(
    nu::Int, nx::Int, nv::Int, ny::Int, ν::Number=T(0), ρ::Number=T(0);
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

    # Check ρ and ν
    if ρ < 0 || ν < 0
        @warn("Warning: negative passivity index detected, passivity is NOT guaranteed")
    end 

    if ρ*ν > 0
        error("If ρ and ν are both positive, passiveREN could produce incorrect results. Please set at least one of them as zero. ")               
    end

    # Direct (implicit) params
    direct_ps = DirectRENParams{T}(
        nu, nx, nv, ny; 
        init, ϵ, bx_scale, bv_scale, polar_param, 
        D22_free=false, rng,
    )

    return PassiveRENParams{T}(nl, nu, nx, nv, ny, direct_ps, αbar, ν, ρ)

end

@functor PassiveRENParams
trainable(m::PassiveRENParams) = (direct = m.direct, )

function direct_to_explicit(ps::PassiveRENParams{T}, return_h=false) where T

    # System sizes
    ν = ps.ν
    ρ = ps.ρ

    # Implicit parameters
    ϵ = ps.direct.ϵ
    ρ_polar = ps.direct.ρ
    X = ps.direct.X
    polar_param = ps.direct.polar_param

    X3 = ps.direct.X3
    Y3 = ps.direct.Y3

    # Implicit system and output matrices
    B2_imp = ps.direct.B2
    D12_imp = ps.direct.D12

    C2 = ps.direct.C2
    D21 = ps.direct.D21

    # Constructing D22 for incrementally (strictly input) passive and incrementally strictly output passive. 
    # See Eqns 31-33 of TAC paper 
    M = _M_pass(X3, Y3, ϵ)

    if ρ == 0
        # For ρ==0 case, I(SI)P model
        D22 = ν*I + M
        D21_imp = D21 - D12_imp'

        𝑅  = _R_pass(D22, ν, ρ) 
        Γ2 = _Γ2_pass(C2, D21_imp, B2_imp, 𝑅)

        H = x_to_h(X, ϵ, polar_param, ρ_polar) + Γ2
    else    
        # For ρ!=0 case, ISOP model
        D22 = _D22_pass(M, ρ) 
        C2_imp = _C2_pass(D22, C2, ρ)
        D21_imp = _D21_pass(D22, D21, D12_imp, ρ)

        𝑅  = _R_pass(D22, ν, ρ)

        Γ1 = _Γ1_pass(ps.nx, ps.ny, C2, D21, ρ, T) 
        Γ2 = _Γ2_pass(C2_imp, D21_imp, B2_imp, 𝑅)

        H = x_to_h(X, ϵ, polar_param, ρ_polar) + Γ2 - Γ1
    end
    
    # Get explicit parameterisation
    !return_h && (return hmatrix_to_explicit(ps, H, D22))
    return H

end

_D22_pass(M, ρ) = (I+M)\I/ρ   

_C2_pass(D22, C2, ρ) = (D22'*(-2ρ*I) + I)*C2

_D21_pass(D22, D21, D12_imp, ρ) = (D22'*(-2ρ*I) + I)*D21 - D12_imp'

_M_pass(X3, Y3, ϵ) = X3'*X3 + Y3 - Y3' + ϵ*I

_R_pass(D22, ν, ρ) = -2ν*I + D22 + D22' + D22'*(-2ρ*I)*D22

function _Γ1_pass(nx, ny, C2, D21, ρ, T) 
    [C2'; D21'; zeros(T, nx, ny)] * (-2ρ*I) * [C2 D21 zeros(T, ny, nx)]
end

function _Γ2_pass(C2, D21_imp, B2_imp, 𝑅)
    [C2'; D21_imp'; B2_imp] * (𝑅 \ [C2 D21_imp B2_imp'])
end