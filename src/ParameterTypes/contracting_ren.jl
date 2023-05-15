mutable struct ContractingRENParams{T} <: AbstractRENParams{T}
    nl::Function                # Sector-bounded nonlinearity
    nu::Int
    nx::Int
    nv::Int
    ny::Int
    direct::DirectRENParams{T}
    αbar::T
end

"""
    ContractingRENParams{T}(nu, nx, nv, ny; <keyword arguments>) where T

Construct direct parameterisation of a contracting REN.

The parameters can be used to construct an explicit [`REN`](@ref) model that has guaranteed, built-in contraction properties.

# Arguments
- `nu::Int`: Number of inputs.
- `nx::Int`: Number of states.
- `nv::Int`: Number of neurons.
- `ny::Int`: Number of outputs.

# Keyword arguments

- `nl::Function=Flux.relu`: Sector-bounded static nonlinearity.

- `αbar::T=1`: Upper bound on the contraction rate with `ᾱ ∈ (0,1]`.

See [`DirectRENParams`](@ref) for documentation of keyword arguments `init`, `ϵ`, `bx_scale`, `bv_scale`, `polar_param`, `D22_zero`, `is_output`, `rng`.

See also [`GeneralRENParams`](@ref), [`LipschitzRENParams`](@ref), [`PassiveRENParams`](@ref).
"""
function ContractingRENParams{T}(
    nu::Int, nx::Int, nv::Int, ny::Int;
    nl::Function        = Flux.relu, 
    αbar::T             = T(1),
    init                = :random,
    polar_param::Bool   = true,
    D22_zero::Bool      = false,
    bx_scale::T         = T(0), 
    bv_scale::T         = T(1), 
    is_output::Bool     = true,
    ϵ::T                = T(1e-12), 
    rng::AbstractRNG    = Random.GLOBAL_RNG
) where T

    # Direct (implicit) params
    direct_ps = DirectRENParams{T}(
        nu, nx, nv, ny; 
        init=init, ϵ=ϵ, bx_scale=bx_scale, bv_scale=bv_scale, 
        polar_param=polar_param, D22_free=true, D22_zero=D22_zero,
        is_output=is_output, rng=rng
    )

    return ContractingRENParams{T}(nl, nu, nx, nv, ny, direct_ps, αbar)

end

@doc raw"""
    ContractingRENParams(nv, A, B, C, D; ...)

Alternative constructor for `ContractingRENParams` that initialises the
REN from a **stable** discrete-time linear system with state-space model

```math
\begin{align*}
x_{t+1} &= Ax_t + Bu_t \\
y_t &= Cx_t + Du_t.
\end{align*}
```

[TODO:] This method may be removed in a later edition of the package.

[TODO:] Make compatible with αbar ≠ 1.0.
"""
function ContractingRENParams(
    nv::Int,
    A::AbstractMatrix{T}, B::AbstractMatrix{T}, C::AbstractMatrix{T}, D::AbstractMatrix{T};
    nl = Flux.relu, 
    ϵ = T(1e-12), 
    bx_scale = T(0), 
    bv_scale = T(1), 
    polar_param = true, 
    df_wishart = nothing,
    rng = Random.GLOBAL_RNG
) where T

    # Check linear system stability
    if maximum(abs.(eigvals(A))) >= 1
        error("System must be stable: spectral radius of A < 1")
    end

    # Get system sizes
    nx = size(A,1)
    nu = size(B,2)
    ny = size(C,1)

    # Use given matrices for explicit REN (\bb for explicit)
    𝔸  = A; 𝔹2  = B
    ℂ2 = C; 𝔻22 = D

    # Set others to zero (so explicit = direct)
    B1 = zeros(T, nx, nv)
    C1 = zeros(T, nv, nx)
    D11 = zeros(T, nv, nv)
    D12 = zeros(T, nv, nu)
    𝔻21 = zeros(T, ny, nv)

    # Solve Lyapunov equation
    P = lyapd(𝔸',I(nx))

    # Implicit system params
    E = P
    Λ = I
    ρ = zeros(T, 1)
    Y1 = zeros(T, nx, nx)

    F = E * 𝔸
    B2 = E * 𝔹2
    H22 = 2Λ - D11 - D11'
    H = [(E + E' - P) -C1' F'; -C1 H22 B1'; F B1 P] + ϵ*I

    # Sample nearby nonlinear systems and decompose
    (df_wishart !== nothing) && (H = rand(Wishart(df_wishart,H))/df_wishart + ϵ*I)
    X = Matrix{T}(cholesky(H).U)

    # Add bias terms
    bv = T(bv_scale) * glorot_normal(nv; T=T, rng=rng)
    bx = T(bx_scale) * glorot_normal(nx; T=T, rng=rng)
    by = glorot_normal(ny; T=T, rng=rng)

    # D22 parameterisation
    D22_free = true
    D22_zero = false
    X3 = zeros(T, 0, 0)
    Y3 = zeros(T, 0, 0)
    Z3 = zeros(T, 0, 0)

    # Build REN params
    αbar = T(1)
    direct_ps = DirectRENParams{T}(
        ρ, X, Y1, X3, Y3, Z3, B2, 
        ℂ2, D12, 𝔻21, 𝔻22, bx, bv, by, 
        ϵ, polar_param, D22_free, D22_zero
    )

    return ContractingRENParams{T}(nl, nu, nx, nv, ny, direct_ps, αbar)

end

Flux.@functor ContractingRENParams (direct, )

function Flux.gpu(m::ContractingRENParams{T}) where T
    # TODO: Test and complete this
    direct_ps = Flux.gpu(m.direct)
    return ContractingRENParams{T}(
        m.nl, m.nu, m.nx, m.nv, m.ny, direct_ps, m.αbar
    )
end

function Flux.cpu(m::ContractingRENParams{T}) where T
    # TODO: Test and complete this
    direct_ps = Flux.cpu(m.direct)
    return ContractingRENParams{T}(
        m.nl, m.nu, m.nx, m.nv, m.ny, direct_ps, m.αbar
    )
end

function direct_to_explicit(ps::ContractingRENParams{T}, return_h::Bool=false) where T

    ϵ = ps.direct.ϵ
    ρ = ps.direct.ρ[1]
    X = ps.direct.X
    polar_param = ps.direct.polar_param
    H = x_to_h(X, ϵ, polar_param, ρ)
    
    !return_h && (return hmatrix_to_explicit(ps, H))
    return H

end