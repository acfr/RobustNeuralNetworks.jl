# This file is a part of RobustNeuralNetworks.jl. License is MIT: https://github.com/acfr/RobustNeuralNetworks.jl/blob/main/LICENSE 

using LinearAlgebra
using Random
using RobustNeuralNetworks
using Test

# include("../test_utils.jl")

"""
Test that the behavioural constraints are satisfied
"""
batches = 42
nu, nx, nv, ny = 10, 5, 10, 20

# Generate random matrices
X = randn(ny,ny)
Y = randn(nu,nu)
S = rand(nu,ny)

Q = -X'*X
R = S * (Q \ S') + Y'*Y

ren_ps = GeneralRENParams{Float64}(nu, nx, nv, ny, Q, S, R)
ren = REN(ren_ps)

# Different inputs with different initial conditions
u0 = 10*randn(nu, batches)
u1 = rand(nu, batches)

x0 = randn(nx, batches)
x1 = randn(nx, batches)

# Simulate
x0n, y0 = ren(x0, u0)
x1n, y1 = ren(x1, u1)

# Test behavioural constraint
M = [Q S'; S R]
rhs = mat_norm2(M, vcat(y1 .- y0, u1 .- u0))

P = compute_p(ren_ps)
lhs = mat_norm2(P, x0n .- x1n) - mat_norm2(P, x0 .- x1)

@test all(lhs .<= rhs)
