# This file is a part of RobustNeuralNetworks.jl. License is MIT: https://github.com/acfr/RobustNeuralNetworks.jl/blob/main/LICENSE 

using Random
using RobustNeuralNetworks
using Test

# include("../test_utils.jl")

rng = MersenneTwister(42)

"""
Test that the contracting REN actually does contract.

Test uses some of the additional options just to double-check.
"""
batches = 42
nu, nx, nv, ny = 4, 5, 10, 5
ᾱ = 0.5

ren_ps = ContractingRENParams{Float64}(
    nu, nx, nv, ny; 
    init=:cholesky, αbar=ᾱ, polar_param=false, output_map=false, rng
)
ren = REN(ren_ps)

# Same inputs. different initial conditions
u0 = randn(rng, nu, batches)

x0 = randn(rng, nx, batches)
x1 = randn(rng, nx, batches)

# Simulate
x0n, y0 = ren(x0, u0)
x1n, y1 = ren(x1, u0)

# Test contraction condition
P = compute_p(ren_ps)
lhs = mat_norm2(P, x0n .- x1n) - mat_norm2(P, x0 .- x1)*ᾱ^2
rhs = 0.0

@test all(lhs .<= rhs)
