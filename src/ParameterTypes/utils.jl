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
function hmatrix_to_explicit(ps::AbstractRENParams, H::AbstractMatrix{T}, D22::AbstractMatrix{T} = zeros(T,0,0)) where T<:Real

    # System sizes
    nx = ps.nx
    nv = ps.nv

    # To be used later
    ᾱ = ps.αbar
    Y1 = ps.direct.Y1
    
    # Extract sections of H matrix 
    # Using @view is faster but not supported by CUDA
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
    Λ_inv = ( (1 ./ diag(H22)) * 2)
    D11_imp = (-tril(H22, -1))

    # Construct the explicit model
    A = E \ F
    B1 = E \ B1_imp
    B2 = E \ ps.direct.B2

    C1 = broadcast(*, Λ_inv, C1_imp)
    D11 = broadcast(*, Λ_inv, D11_imp)
    D12 = broadcast(*, Λ_inv, ps.direct.D12)

    C2 = ps.direct.C2
    D21 = ps.direct.D21

    isempty(D22) && (D22 = ps.direct.D22)

    bx = ps.direct.bx
    bv = ps.direct.bv
    by = ps.direct.by
    
    return ExplicitRENParams{T}(A, B1, B2, C1, C2, D11, D12, D21, D22, bx, bv, by)

end

function x_to_h(X::AbstractMatrix{T}, ϵ::T, polar_param::Bool, ρ::T) where T
    polar_param ? (ρ^2)*(X'*X) / norm(X)^2 + ϵ*I : X'*X + ϵ*I
end