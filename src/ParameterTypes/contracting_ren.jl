"""
$(TYPEDEF)

Parameter struct to build a contracting acyclic REN.
"""
mutable struct ContractingRENParams{T} <: AbstractRENParams{T}
    nl                          # Sector-bounded nonlinearity
    nu::Int
    nx::Int
    nv::Int
    ny::Int
    direct::DirectParams{T}
    output::OutputLayer{T}
    αbar::T
end

"""
    ContractingRENParams(nu, nx, nv, ny; ...)

Main constructor for `ContractingRENParams`.
ᾱ ∈ (0,1] is the upper bound on contraction rate.
"""
function ContractingRENParams{T}(
    nu::Int, nx::Int, nv::Int, ny::Int;
    init = :random,
    nl = Flux.relu, 
    ϵ = T(0.001), 
    αbar = T(1),
    bx_scale = T(0), 
    bv_scale = T(1), 
    polar_param = true,
    rng = Random.GLOBAL_RNG
) where T

    # Direct (implicit) params
    direct_ps = DirectParams{T}(
        nu, nx, nv, ny; 
        init=init, ϵ=ϵ, bx_scale=bx_scale, bv_scale=bv_scale, 
        polar_param=polar_param, D22_free=true, rng=rng
    )

    # Output layer
    output_ps = OutputLayer{T}(nu, nx, nv, ny; D22_trainable=true, rng=rng)

    return ContractingRENParams{T}(nl, nu, nx, nv, ny, direct_ps, output_ps, αbar)

end

"""
    ContractingRENParams(nv, A, B, C, D; ...)

Alternative constructor for `ContractingRENParams` that initialises the
REN from a **stable** discrete-time linear system ss(A,B,C,D).

TODO: Make compatible with αbar ≠ 1.0
"""
function ContractingRENParams(
    nv::Int,
    A::AbstractMatrix{T}, B::AbstractMatrix{T}, C::AbstractMatrix{T}, D::AbstractMatrix{T};
    nl = Flux.relu, 
    ϵ = T(1e-6), 
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
    V = Matrix{T}(cholesky(H).U)

    # Add bias terms
    bv = T(bv_scale) * glorot_normal(nv; T=T, rng=rng)
    bx = T(bx_scale) * glorot_normal(nx; T=T, rng=rng)
    by = glorot_normal(ny; T=T, rng=rng)

    # D22 parameterisation
    D22_free = true
    D22_trainable = true
    X3 = zeros(T, 0, 0)
    Y3 = zeros(T, 0, 0)
    Z3 = zeros(T, 0, 0)

    # Build REN params
    αbar = T(1)
    direct_ps = DirectParams{T}(ρ, V, Y1, X3, Y3, Z3, B2, D12, bx, bv, ϵ, polar_param, D22_free)
    output_ps = OutputLayer{T}(ℂ2, 𝔻21, 𝔻22, by, D22_trainable)

    return ContractingRENParams{T}(nl, nu, nx, nv, ny, direct_ps, output_ps, αbar)

end

"""
    Flux.trainable(m::ContractingRENParams)

Define trainable parameters for `ContractingRENParams` type
Filter empty ones (handy when nx=0)
"""
Flux.trainable(m::ContractingRENParams) = filter(
    p -> length(p) !=0, 
    (Flux.trainable(m.direct)..., Flux.trainable(m.output)...)
)

"""
    Flux.gpu(m::ContractingRENParams{T}) where T

Add GPU compatibility for `ContractingRENParams` type
"""
function Flux.gpu(m::ContractingRENParams{T}) where T
    direct_ps = Flux.gpu(m.direct)
    output_ps = Flux.gpo(m.output)
    return ContractingRENParams{T}(
        m.nl, m.nu, m.nx, m.nv, m.ny, direct_ps, output_ps, m.αbar
    )
end

"""
    Flux.cpu(m::ContractingRENParams{T}) where T

Add CPU compatibility for `ContractingRENParams` type
"""
function Flux.cpu(m::ContractingRENParams{T}) where T
    direct_ps = Flux.cpu(m.direct)
    output_ps = Flux.cpo(m.output)
    return ContractingRENParams{T}(
        m.nl, m.nu, m.nx, m.nv, m.ny, direct_ps, output_ps, m.αbar
    )
end

"""
    direct_to_explicit(ps::ContractingRENParams)

Convert direct REN parameterisation to explicit parameterisation
for contracting REN
"""
function direct_to_explicit(ps::ContractingRENParams{T}) where T

    # System sizes
    nx = ps.nx
    nv = ps.nv

    # To be used later
    ᾱ = ps.αbar
    Y1 = ps.direct.Y1

    # Extract useful parameters and construct H
    ϵ = ps.direct.ϵ
    ρ = ps.direct.ρ
    V = ps.direct.V
    H = ps.direct.polar_param ? exp(ρ[1])*V'*V / norm(V)^2 : V'*V + ϵ*I

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
    D12 = Λ_inv .* ps.direct.D12

    C2 = ps.output.C2
    D21 = ps.output.D21
    D22 = ps.output.D22

    bx = ps.direct.bx
    bv = ps.direct.bv
    by = ps.output.by
    
    return ExplicitParams{T}(A, B1, B2, C1, C2, D11, D12, D21, D22, bx, bv, by)

end