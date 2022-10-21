using LinearAlgebra
using Random
using RecurrentEquilibriumNetworks
using Test

"""
Test that the behavioural constraints are satisfied
"""
batches = 100
nu, nx, nv, ny = 10, 5, 10, 20

# Generate random matrices
X = randn(ny,ny)
Y = randn(nu,nu)
S = rand(nu,ny)

Q = -X'*X
R = S * (Q \ S') + Y'*Y

ren_ps = GeneralRENParams{Float64}(nu, nx, nv, ny, Q, S, R)
ren = REN(ren_ps)

# Different inputs with same initial condition
u0 = 10*randn(nu, batches)
u1 = rand(nu, batches)

x0 = randn(nx, batches)

# Simulate
_, y0 = ren(x0, u0)
_, y1 = ren(x0, u1)

# Test behavioural constraint
dyu = vcat(y1 .- y0, u1 .- u0)
M = [Q S'; S R]
condition = sum(dyu .* (M * dyu); dims=1)

@test all(condition .>= 0)
