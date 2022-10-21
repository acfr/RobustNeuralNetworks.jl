using Random
using RecurrentEquilibriumNetworks
using Test


"""
Test that the model satisfies a specified Lipschitz bound
"""
batches = 100
nu, nx, nv, ny = 4, 5, 10, 2
γ = 10

ren_ps = LipschitzRENParams{Float64}(nu, nx, nv, ny, γ)
ren = REN(ren_ps)

# Different inputs with same initial condition
u0 = 10*randn(nu, batches)
u1 = rand(nu, batches)

x0 = randn(nx, batches)

# Simulate
_, y0 = ren(x0, u0)
_, y1 = ren(x0, u1)

# Test Lipschitz condition
vecnorm(A,d=1) = sqrt.(sum(abs.(A .^2); dims=d))

norm_dy = vecnorm(y0 - y1)
norm_du = vecnorm(u0 - u1)

@test all(norm_dy .<= γ * norm_du)
