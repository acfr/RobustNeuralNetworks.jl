"""
    hmatrix_to_explicit(ps::AbstractRENParams, H::Matrix{T}, D22::Matrix{T} = zeros(T,0,0)) where T

Convert direct REN parameterisation encoded in H matrix
to explicit parameterisation. See TAC paper for details
"""
function hmatrix_to_explicit(ps::AbstractRENParams, H::Matrix{T}, D22::Matrix{T} = zeros(T,0,0)) where T

    # System sizes
    nx = ps.nx
    nv = ps.nv

    # To be used later
    ᾱ = ps.αbar
    Y1 = ps.direct.Y1
    
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

    C2 = ps.direct.C2
    D21 = ps.direct.D21
    isempty(D22) && (D22 = ps.direct.D22)

    bx = ps.direct.bx
    bv = ps.direct.bv
    by = ps.direct.by
    
    return ExplicitParams{T}(A, B1, B2, C1, C2, D11, D12, D21, D22, bx, bv, by)

end