# This file is a part of RobustNeuralNetworks.jl. License is MIT: https://github.com/acfr/RobustNeuralNetworks.jl/blob/main/LICENSE 

using Random
using RobustNeuralNetworks
using Test

# include("../test_utils.jl")

rng = MersenneTwister(42)

"""
Test that the model satisfies a specified Lipschitz bound
"""
batches = 100
nu, ny = 4, 2
nh = [10, 5, 20, 4]
γ = 1e-5

lbdn_ps = DenseLBDNParams{Float64}(nu, nh, ny, γ; rng)
lbdn = LBDN(lbdn_ps)

# Different inputs with different initial conditions
u0 = randn(rng, nu, batches)
u1 = u0 .+ 0.001*rand(rng, nu, batches)

# Simulate
y0 = lbdn(u0)
y1 = lbdn(u1)

# Test Lipschitz condition
lhs = 0
rhs = γ^2 * vecnorm2(u0 - u1) - vecnorm2(y0 - y1)

@test all(lhs .<= rhs)