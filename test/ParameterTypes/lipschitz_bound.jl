# This file is a part of RobustNeuralNetworks.jl. License is MIT: https://github.com/acfr/RobustNeuralNetworks.jl/blob/main/LICENSE 

using Random
using RobustNeuralNetworks
using Test

# include("../test_utils.jl")

rng = MersenneTwister(42)

"""
Test that the model satisfies a specified Lipschitz bound
"""
batches = 42
nu, nx, nv, ny = 4, 5, 10, 2
γ = 10

ren_ps = LipschitzRENParams{Float64}(nu, nx, nv, ny, γ; rng)
ren = REN(ren_ps)

# Different inputs with different initial conditions
u0 = 10*randn(rng, nu, batches)
u1 = rand(rng, nu, batches)

x0 = randn(rng, nx, batches)
x1 = randn(rng, nx, batches)

# Simulate
x0n, y0 = ren(x0, u0)
x1n, y1 = ren(x1, u1)

# Test Lipschitz condition
P = compute_p(ren_ps)
lhs = mat_norm2(P, x0n .- x1n) - mat_norm2(P, x0 .- x1)
rhs = γ^2 * vecnorm2(u0 - u1) - vecnorm2(y0 - y1)

@test all(lhs .<= rhs)
