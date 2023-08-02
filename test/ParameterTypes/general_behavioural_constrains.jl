# This file is a part of RobustNeuralNetworks.jl. License is MIT: https://github.com/acfr/RobustNeuralNetworks.jl/blob/main/LICENSE 

using LinearAlgebra
using Random
using RobustNeuralNetworks
using Test

# include("../test_utils.jl")

rng = MersenneTwister(42)

"""
Test that the behavioural constraints are satisfied
"""
batches = 42
nu, nx, nv, ny = 10, 5, 10, 20

# Generate random matrices
X = randn(rng, ny,ny)
Y = randn(rng, nu,nu)
S = rand(rng, nu,ny)

Q = -X'*X
R = S * (Q \ S') + Y'*Y

ren_ps = GeneralRENParams{Float64}(nu, nx, nv, ny, Q, S, R; rng)
ren = REN(ren_ps)

# Different inputs with different initial conditions
u0 = 10*randn(rng, nu, batches)
u1 = rand(rng, nu, batches)

x0 = randn(rng, nx, batches)
x1 = randn(rng, nx, batches)

# Simulate
x0n, y0 = ren(x0, u0)
x1n, y1 = ren(x1, u1)

# Test behavioural constraint
M = [Q S'; S R]
rhs = mat_norm2(M, vcat(y1 .- y0, u1 .- u0))

P = compute_p(ren_ps)
lhs = mat_norm2(P, x0n .- x1n) - mat_norm2(P, x0 .- x1)

@test all(lhs .<= rhs)
