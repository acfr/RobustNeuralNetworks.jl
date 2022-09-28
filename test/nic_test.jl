cd(@__DIR__)
using Pkg
Pkg.activate("./..")

using BenchmarkTools
using LinearAlgebra
using RecurrentEquilibriumNetworks

"""
Nic's random testing script
"""

nu = 4
nx = 10
nv = 20
ny = 2

# Test constructors
params = DirectParams{Float64}(nu, nx, nv, ny; init=:random)
params = DirectParams{Float32}(nu, nx, nv, ny; init=:cholesky)

cren_ps = ContractingRENParams{Float64}(nu, nx, nv, ny; init=:random)

# Test linear system initialisation for fun
A = Matrix{Float64}(I(nx)*0.99)
B = randn(nx,nu)
C = randn(ny,nx)
D = randn(ny,nu)

cren_ps_lsys = ContractingRENParams(nv, A, B, C, D)

# Test out the REN class
function test_ren(ren_ps, batches = 20)

    ren = REN(ren_ps)

    x0 = init_states(ren)
    x0s = init_states(ren, batches)

    u0 = randn(ren.nu)
    u0s = randn(ren.nu, batches)

    x1, y1 = ren(x0, u0)
    x1s, y1s = ren(x0s, u0s)

    return nothing

end

test_ren(cren_ps)

# Test more general REN construction with Q,S,R matrices

println("Made it to the end")
# exit()

# Things to test:
#   - Various constructions with D22 trainable/free or not (4 combinations)
#   - GeneralRENParams (everything)
#   - CPU/GPU compatibility
