# This file is a part of RobustNeuralNetworks.jl. License is MIT: https://github.com/acfr/RobustNeuralNetworks.jl/blob/main/LICENSE 

using LinearAlgebra
using Random
using RobustNeuralNetworks
using Test

# include("../test_utils.jl")

rng = MersenneTwister(42)

"""
Test passivity inequality
"""
batches = 100
nu, nx, nv, ny = 6, 5, 10, 6
T = 100

# Test constructors
ren_ps = PassiveRENParams{Float64}(nu, nx, nv, ny; init=:random, ν= 1.0, rng)
ren = REN(ren_ps)

# Different inputs with different initial conditions
u0 = 10*randn(rng, nu, batches)
u1 = rand(rng, nu, batches)

x0 = randn(rng, nx, batches)
x1 = randn(rng, nx, batches)

# Simulate
x0n, y0 = ren(x0, u0)
x1n, y1 = ren(x1, u1)

# Dissipation condition
ν = ren_ps.ν
Q = zeros(ny, ny)
S = Matrix(I, nu, ny)
R = -2ν * Matrix(I, nu, nu)

# Test passivity
M = [Q S'; S R]
rhs = mat_norm2(M, vcat(y1 .- y0, u1 .- u0))

P = compute_p(ren_ps)
lhs = mat_norm2(P, x0n .- x1n) - mat_norm2(P, x0 .- x1)

@test all(lhs .<= rhs)
