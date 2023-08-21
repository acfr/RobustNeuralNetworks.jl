# This file is a part of RobustNeuralNetworks.jl. License is MIT: https://github.com/acfr/RobustNeuralNetworks.jl/blob/main/LICENSE 

# Backprop through I + Z was performing scalar indexing on GPU
# Ensuring I is the same type as Z avoids this
_get_I(Z::T) where T = T(I(size(Z,1)))
@non_differentiable _get_I(Z)

"""
    direct_to_explicit(ps::AbstractRENParams{T}, return_h=false) where T

Convert direct parameterisation of RENs to explicit parameterisation.

Uses the parameterisation encoded in `ps` to construct an [`ExplicitRENParams`](@ref) object that naturally satisfies a set of user-defined behavioural constraints.

# Arguments

- `ps::AbstractRENParams`: Direct parameterisation with behavioural constraints to convert to an explicit parameterisation of REN (eg: [`GeneralRENParams`](@ref)).

- `return_h::Bool=false`: Whether to return the H-matrix directly (see [Revay et al. (2021)](https://arxiv.org/abs/2104.05942)). Useful for debugging or model analysis. If `false`, function returns an object of type `ExplicitRENParams{T}`. 

See also [`GeneralRENParams`](@ref), [`ContractingRENParams`](@ref), [`LipschitzRENParams`](@ref), [`PassiveRENParams`](@ref).
"""
function direct_to_explicit(ps::AbstractRENParams{T}, return_h=false) where T end

"""
    direct_to_explicit(ps::AbstractRENParams{T}) where T

Convert direct parameterisation of LBDNs to explicit parameterisation.

Uses the parameterisation encoded in `ps` to construct an [`ExplicitLBDNParams`](@ref) object that naturally respects a user-defined Lipschitz bound.

# Arguments

- `ps::AbstractLBDNParams`: Direct parameterisation of an LBDN to convert to an explicit parameterisation for model evaluation (eg: [`DenseLBDNParams`](@ref)).

See also [`DenseLBDNParams`](@ref).
"""
function direct_to_explicit(ps::AbstractLBDNParams{T}) where T end

"""
    hmatrix_to_explicit(ps, H, D22=zeros(T,0,0)) where T

Convert direct parameterisation of REN from H matrix (Eqn. 23 of [Revay et al. (2021)](https://arxiv.org/abs/2104.05942)) to `ExplicitRENParams{T}`.

# Arguments
- `ps::AbstractRENParams`: Direct parameterisation of a REN with behavioural constraints
- `H::Matrix{T}`: H-matrix to convert.
- `D22::Matrix{T}=zeros(T,0,0))`: Optionally include `D22` matrix. If empty (default), `D22` taken from `ps.direct.D22`. 
"""
function hmatrix_to_explicit(ps::AbstractRENParams, H::AbstractMatrix{T}, D22::AbstractMatrix{T} = zeros(T,0,0)) where T

    # System sizes
    nx = ps.nx
    nv = ps.nv
    nu = ps.nu

    # To be used later
    ᾱ = ps.αbar
    Y1 = ps.direct.Y1
    
    # Extract sections of H matrix 
    H11 = H[1:nx, 1:nx]
    H22 = H[nx + 1:nx + nv, nx + 1:nx + nv]
    H33 = H[nx + nv + 1:2nx + nv, nx + nv + 1:2nx + nv]
    H21 = H[nx + 1:nx + nv, 1:nx]
    H31 = H[nx + nv + 1:2nx + nv, 1:nx]
    H32 = H[nx + nv + 1:2nx + nv, nx + 1:nx + nv]

    # Construct implicit model parameters
    P_imp = H33
    F = H31
    E = _E(H11, P_imp, ᾱ, Y1)

    # Equilibrium network parameters
    B1_imp = H32
    C1_imp = -H21
    Λ_inv = _Λ_inv(H22)
    D11_imp = (-tril(H22, -1))

    # Construct the explicit model
    A = E \ F
    B1 = E \ B1_imp
    B2 = E \ ps.direct.B2

    # Equilibrium layer matrices
    C1  = _C1(nv, nx, Λ_inv, C1_imp, T)
    D11 = _D11(nv, Λ_inv, D11_imp, T)
    D12 = _D12(nv, nu, Λ_inv, ps.direct.D12, T)

    # Output layer
    C2 = ps.direct.C2
    D21 = ps.direct.D21
    isempty(D22) && (D22 = ps.direct.D22)

    # Biases
    bx = ps.direct.bx
    bv = ps.direct.bv
    by = ps.direct.by
    
    return ExplicitRENParams{T}(A, B1, B2, C1, C2, D11, D12, D21, D22, bx, bv, by)
end

# Splitting operations into functions speeds up auto-diff
_E(H11, P_imp, ᾱ, Y1) = (H11 + P_imp/ᾱ^2 + Y1 - Y1')/2
_Λ_inv(H22) = ( (1 ./ diag(H22)) * 2)

# Current versions of Julia behave poorly when broadcasting over 0-dim arrays
_C1(nv, nx, Λ_inv, C1_imp, T)   = (nv == 0) ? zeros(T,0,nx) : broadcast(*, Λ_inv, C1_imp)
_D11(nv, Λ_inv, D11_imp, T)     = (nv == 0) ? zeros(T,0,0)  : broadcast(*, Λ_inv, D11_imp)
_D12(nv, nu, Λ_inv, D12_imp, T) = (nv == 0) ? zeros(T,0,nu) : broadcast(*, Λ_inv, D12_imp)

x_to_h(X, ϵ, polar_param, ρ) = polar_param ? (ρ.^2).*(X'*X) / norm(X)^2 + ϵ*I : X'*X + ϵ*I

"""
    set_output_zero!(m::AbstractRENParams)

Set output map of a REN to zero.

If the resulting model is called with
```julia
ren = REN(m)
x1, y = ren(x, u)
```
then `y = 0` for any `x` and `u`.
"""
function set_output_zero!(m::AbstractRENParams)
    m.direct.C2  .= 0
    m.direct.D21 .= 0
    m.direct.D22 .= 0
    m.direct.by  .= 0
    return nothing
end

"""
    set_output_zero!(m::AbstractLBDNParams)

Set output map of an LBDN to zero.

If the resulting model is called with 
```julia
lbdn = LBDN(m)
y = lbdn(u)
```
then `y = 0` for any `u`.
"""
function set_output_zero!(m::AbstractLBDNParams)
    m.direct.XY[end][(m.ny+1):end,:] .= 0
    m.direct.b[end] .= 0

    return nothing
end