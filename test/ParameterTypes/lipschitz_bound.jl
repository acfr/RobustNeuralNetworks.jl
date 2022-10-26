using Random
using RecurrentEquilibriumNetworks
using Test

# include("../test_utils.jl")

"""
Test that the model satisfies a specified Lipschitz bound
"""
batches = 100
nu, nx, nv, ny = 4, 5, 10, 2
γ = 10

ren_ps = LipschitzRENParams{Float64}(nu, nx, nv, ny, γ)
ren = REN(ren_ps)

# Different inputs with different initial conditions
u0 = 10*randn(nu, batches)
u1 = rand(nu, batches)

x0 = randn(nx, batches)
x1 = randn(nx, batches)

# Simulate
x0n, y0 = ren(x0, u0)
x1n, y1 = ren(x1, u1)

# Test Lipschitz condition
vecnorm2(A,d=1) = sum(abs.(A .^2); dims=d)

P = compute_p(ren_ps)
lhs = mat_norm2(P, x0n .- x1n) - mat_norm2(P, x0 .- x1)
rhs = γ^2 * vecnorm2(u0 - u1) - vecnorm2(y0 - y1)

@test all(lhs .<= rhs)
