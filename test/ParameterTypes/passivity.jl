using LinearAlgebra
using Random
using RecurrentEquilibriumNetworks
using Test

"""
Test passivity inequality
"""
batches = 100
nu, nx, nv, ny = 6, 5, 10, 6
T = 100

# Test constructors
Params = PassiveRENParams{Float64}(nu, nx, nv, ny; init=:random, ν= 1.0)
ren = REN(Params)

# Different inputs with same initial condition
u0 = 10*randn(nu, batches)
u1 = rand(nu, batches)

x0 = randn(nx, batches)

# Simulate
_, y0 = ren(x0, u0)
_, y1 = ren(x0, u1)

# Test passivity
dyu = vcat(y1 .- y0, u1 .- u0)

# Dissipation condition
ν = Params.ν
Q = zeros(ny, ny)
S = Matrix(I, nu, ny)
R = -2ν * Matrix(I, nu, nu)
M = [Q S'; S R]
condition = sum(dyu .* (M * dyu); dims=1)

@test all(condition .>= 0)
