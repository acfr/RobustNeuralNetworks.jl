# This file is a part of RobustNeuralNetworks.jl. License is MIT: https://github.com/acfr/RobustNeuralNetworks.jl/blob/main/LICENSE 

mutable struct GeneralRENParams{T} <: AbstractRENParams{T}
    nl::Function                # Sector-bounded nonlinearity
    nu::Int
    nx::Int
    nv::Int
    ny::Int
    direct::DirectRENParams{T}
    αbar::T
    Q::Matrix{T}
    S::Matrix{T}
    R::Matrix{T}
end

"""
    GeneralRENParams{T}(nu, nx, nv, ny, Q, S, R; <keyword arguments>) where T

Construct direct parameterisation of a REN satisfying general behavioural constraints.

Behavioural constraints are encoded by the matrices `Q,S,R` in an incremental Integral Quadratic Constraint (IQC). See Equation 4 of [Revay et al. (2021)](https://arxiv.org/abs/2104.05942).

# Arguments
- `nu::Int`: Number of inputs.
- `nx::Int`: Number of states.
- `nv::Int`: Number of neurons.
- `ny::Int`: Number of outputs.
- `Q::Matrix{T}`: IQC weight matrix on model outputs
- `S::Matrix{T}`: IQC coupling matrix on model outputs/inputs
- `R::Matrix{T}`: IQC weight matrix on model outputs
    
# Keyword arguments

- `nl::Function=relu`: Sector-bounded static nonlinearity.

- `αbar::T=1`: Upper bound on the contraction rate with `ᾱ ∈ (0,1]`.

See [`DirectRENParams`](@ref) for documentation of keyword arguments `init`, `ϵ`, `bx_scale`, `bv_scale`, `polar_param`, `rng`.

See also [`ContractingRENParams`](@ref), [`LipschitzRENParams`](@ref), [`PassiveRENParams`](@ref).
"""
function GeneralRENParams{T}(
    nu::Int, nx::Int, nv::Int, ny::Int,
    Q::Matrix{T}, S::Matrix{T}, R::Matrix{T};
    nl::Function      = relu, 
    αbar::T           = T(1),
    init              = :random,
    polar_param::Bool = true,
    bx_scale::T       = T(0), 
    bv_scale::T       = T(1), 
    ϵ::T              = T(1e-12), 
    rng::AbstractRNG  = Random.GLOBAL_RNG
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
    direct_ps = DirectRENParams{T}(
        nu, nx, nv, ny; 
        init, ϵ, bx_scale, bv_scale, polar_param, 
        D22_free=false, rng,
    )

    return GeneralRENParams{T}(nl, nu, nx, nv, ny, direct_ps, αbar, Q, S, R)

end

@functor GeneralRENParams (direct, )

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
    ρ = ps.direct.ρ[1]
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
    # Currently converts to Hermitian to avoid numerical conditioning issues
    LQ = Matrix{T}(cholesky(-Q).U)
    R1 = Hermitian(R - S * (Q \ S'))
    LR = Matrix{T}(cholesky(R1).U) 
    
    M = _M_gen(X3, Y3, Z3, ϵ)
    N = _N_gen(nu, ny, M, Z3) 

    D22 = _D22_gen(Q, S, LQ, LR, N)

    # Constructing H. See Eqn 28 of TAC paper
    C2_imp = _C2_gen(D22, C2, Q, S)
    D21_imp = _D21_gen(D22, D21, D12_imp, Q, S)

    𝑅  = _R_gen(R, S, Q, D22)
    Γ1 = _Γ1_gen(nx, ny, C2, D21, Q, T) 
    Γ2 = _Γ2_gen(C2_imp, D21_imp, B2_imp, 𝑅)

    H = x_to_h(X, ϵ, polar_param, ρ) + Γ2 - Γ1

    # Get explicit parameterisation
    !return_h && (return hmatrix_to_explicit(ps, H, D22))
    return H

end

# Auto-diff faster through smaller functions
_M_gen(X3, Y3, Z3, ϵ) = X3'*X3 + Y3 - Y3' + Z3'*Z3 + ϵ*I

function _N_gen(nu, ny, M, Z3)
    if ny >= nu
        return[(I - M) / (I + M); -2*Z3 / (I + M)]
    else
        return [((I + M) \ (I - M)) (-2*(I + M) \ Z3')]
    end
end

_D22_gen(Q, S, LQ, LR, N) = -(Q \ S') + (LQ \ N) * LR

_C2_gen(D22, C2, Q, S) = (D22'*Q + S)*C2

_D21_gen(D22, D21, D12_imp, Q, S) = (D22'*Q + S)*D21 - D12_imp'

_R_gen(R, S, Q, D22) = R + S*D22 + D22'*S' + D22'*Q*D22

function _Γ1_gen(nx, ny, C2, D21, Q, T) 
    [C2'; D21'; zeros(T, nx, ny)] * Q * [C2 D21 zeros(T, ny, nx)]
end

function _Γ2_gen(C2_imp, D21_imp, B2_imp, 𝑅)
    [C2_imp'; D21_imp'; B2_imp] * (𝑅 \ [C2_imp D21_imp B2_imp'])
end
