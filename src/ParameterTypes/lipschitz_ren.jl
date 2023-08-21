# This file is a part of RobustNeuralNetworks.jl. License is MIT: https://github.com/acfr/RobustNeuralNetworks.jl/blob/main/LICENSE 

mutable struct LipschitzRENParams{T} <: AbstractRENParams{T}
    nl::Function                # Sector-bounded nonlinearity
    nu::Int
    nx::Int
    nv::Int
    ny::Int
    direct::DirectRENParams{T}
    αbar::T
    γ::AbstractVector{T}
    learn_γ::Bool
end

"""
    LipschitzRENParams(nu, nx, nv, ny, γ; <keyword arguments>) where T

Construct direct parameterisation of a REN with a Lipschitz bound of γ.

# Arguments
- `nu::Int`: Number of inputs.
- `nx::Int`: Number of states.
- `nv::Int`: Number of neurons.
- `ny::Int`: Number of outputs.
- `γ::Number`: Lipschitz upper bound.
    
# Keyword arguments

- `nl::Function=relu`: Sector-bounded static nonlinearity.

- `αbar::T=1`: Upper bound on the contraction rate with `ᾱ ∈ (0,1]`.

- `learn_γ::Bool=false:` Whether to make the Lipschitz bound γ a learnable parameter.

See [`DirectRENParams`](@ref) for documentation of keyword arguments `init`, `ϵ`, `bx_scale`, `bv_scale`, `polar_param`, `D22_zero`, `rng`.

See also [`GeneralRENParams`](@ref), [`ContractingRENParams`](@ref), [`PassiveRENParams`](@ref).
"""
function LipschitzRENParams{T}(
    nu::Int, nx::Int, nv::Int, ny::Int, γ::Number;
    nl::Function      = relu, 
    αbar::T           = T(1),
    learn_γ::Bool     = false,
    init              = :random,
    polar_param::Bool = true,
    bx_scale::T       = T(0), 
    bv_scale::T       = T(1), 
    ϵ::T              = T(1e-12), 
    D22_zero          = false,
    rng::AbstractRNG  = Random.GLOBAL_RNG
) where T

    # If D22 fixed at 0, it should not be constructed from other
    # direct params (so set D22_free = true)
    D22_free = D22_zero ? true : false

    # Direct (implicit) params
    direct_ps = DirectRENParams{T}(
        nu, nx, nv, ny; 
        init, ϵ, bx_scale, bv_scale, polar_param, 
        D22_free, D22_zero, rng,
    )

    return LipschitzRENParams{T}(nl, nu, nx, nv, ny, direct_ps, αbar, [T(γ)], learn_γ)

end

@functor LipschitzRENParams
function trainable(m::LipschitzRENParams)
    m.learn_γ ? (direct = m.direct, γ = m.γ) : (direct = m.direct,)
end

function direct_to_explicit(ps::LipschitzRENParams{T}, return_h=false) where T

    # System sizes
    nu = ps.nu
    nx = ps.nx
    ny = ps.ny

    # Lipschitz bound
    γ = ps.γ

    # Implicit parameters
    ϵ = ps.direct.ϵ
    ρ = ps.direct.ρ
    X = ps.direct.X
    polar_param = ps.direct.polar_param

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
        M = _M_lip(X3, Y3, Z3, ϵ)
        N = _N_lip(nu, ny, M, Z3)
        D22 = γ .* N
    end

    # Constructing H. See Eqn 28 of TAC paper
    C2_imp  = _C2_lip(D22, C2, γ)
    D21_imp = _D21_lip(D22, D21, γ, D12_imp)

    𝑅  = _R_lip(D22, γ)
    Γ1 = _Γ1_lip(nx, ny, C2, D21, γ, T) 
    Γ2 = _Γ2_lip(C2_imp, D21_imp, B2_imp, 𝑅)

    H = x_to_h(X, ϵ, polar_param, ρ) + Γ2 - Γ1

    # Get explicit parameterisation
    !return_h && (return hmatrix_to_explicit(ps, H, D22))
    return H
end

# Auto-diff faster through smaller functions
_M_lip(X3, Y3, Z3, ϵ) = X3'*X3 + Y3 - Y3' + Z3'*Z3 + ϵ*I

function _N_lip(nu, ny, M, Z3)
    Im = _get_I(M) # Prevents scalar indexing on backwards pass of A / (I + M) on GPU
    if ny == nu
        return [(Im + M) \ (Im - M); Z3] # Separate to avoid numerical issues on GPU
    elseif ny >= nu
        return [(Im - M) / (Im + M); -2*Z3 / (Im + M)]
    else
        return [((Im + M) \ (Im - M)) (-2*(Im + M) \ Z3')]
    end
end

_C2_lip(D22, C2, γ) = -(D22')*C2 ./ γ

_D21_lip(D22, D21, γ, D12_imp) = -(D22')*D21 ./ γ - D12_imp'

_R_lip(D22, γ) = γ .* (-D22'*D22 ./ (γ.^2) + I)

function _Γ1_lip(nx, ny, C2, D21, γ, T)
    [C2'; D21'; zeros(T, nx, ny)] * [C2 D21 zeros(T, ny, nx)] .* (-1 ./ γ)
end

function _Γ2_lip(C2_imp, D21_imp, B2_imp, 𝑅)
    [C2_imp'; D21_imp'; B2_imp] * (𝑅 \ [C2_imp D21_imp B2_imp'])
end
