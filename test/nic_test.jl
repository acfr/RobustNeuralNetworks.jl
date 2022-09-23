cd(@__DIR__)
using Pkg
Pkg.activate("./..")

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

cren = ContractingRENParams{Float64}(nu, nx, nv, ny; init=:random)

# Test linear system initialisation for fun
A = Matrix{Float64}(I(nx)*0.99)
B = randn(nx,nu)
C = randn(ny,nx)
D = randn(ny,nu)

cren = ContractingRENParams(nv, A, B, C, D)

e = ExplicitParams{Float64}(
    [1.0 1.0],
    [1.0 1.0],
    [1.0 1.0],
    [1.0 1.0],
    [1.0 1.0],
    [1.0 1.0],
    [1.0 1.0],
    [1.0 1.0],
    [1.0 1.0],
    [1],
    [1],
    [1]
)

println("Made it to the end")

exit()

# Things to test:
#   - Various constructions with D22 trainable/free or not (4 combinations)
#   - GeneralRENParams (everything)
#   - CPU/GPU compatibility
#   - Check that REN() class actually works
