# This file is a part of RobustNeuralNetworks.jl. License is MIT: https://github.com/acfr/RobustNeuralNetworks.jl/blob/main/LICENSE 

"""
Compute P for any REN
"""
function compute_p(ps::AbstractRENParams)

    # System sizes
    nx = ps.nx
    nv = ps.nv

    # To be used later
    ᾱ = ps.αbar
    Y1 = ps.direct.Y1
    
    # Extract sections of H matrix 
    H = direct_to_explicit(ps, true)
    H11 = H[1:nx, 1:nx]
    H33 = H[nx + nv + 1:2nx + nv, nx + nv + 1:2nx + nv]

    # Construct implicit model parameters
    P_imp = H33
    E = (H11 + P_imp/ᾱ^2 + Y1 - Y1')/2

    return E' * (P_imp \ E)
end

"""
Matrix weighted norm
"""
vecnorm2(A,d=1) = sum(abs.(A .^2); dims=d)
mat_norm2(A, x) = sum(x .* (A * x); dims=1)