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
ν = 2.0
ρ = 1.0
T = 100

# Test constructors
ISIPren_ps = PassiveRENParams{Float64}(nu, nx, nv, ny, ν, 0; init=:random, rng)
ISIPren = REN(ISIPren_ps)

ISOPren_ps = PassiveRENParams{Float64}(nu, nx, nv, ny, 0, ρ; init=:random, rng)
ISOPren = REN(ISOPren_ps)

# Different inputs with different initial conditions
u0 = 10*randn(rng, nu, batches)
u1 = rand(rng, nu, batches)

x0 = randn(rng, nx, batches)
x1 = randn(rng, nx, batches)

# Simulate
x0n_ISIP, y0_ISIP = ISIPren(x0, u0)
x1n_ISIP, y1_ISIP = ISIPren(x1, u1)

x0n_ISOP, y0_ISOP = ISOPren(x0, u0)
x1n_ISOP, y1_ISOP = ISOPren(x1, u1)

# Dissipation condition
ν = ISIPren_ps.ν
ρ = ISOPren_ps.ρ
Q_ISIP = zeros(ny, ny)
R_ISIP = -2ν * Matrix(I, nu, nu)
Q_ISOP = -2ρ * Matrix(I, ny, ny)
R_ISOP = zeros(nu, nu)
S = Matrix(I, nu, ny)

# Test passivity
M_ISIP = [Q_ISIP S'; S R_ISIP]
M_ISOP = [Q_ISOP S'; S R_ISOP]

# ISIP case
rhs_ISIP = mat_norm2(M_ISIP, vcat(y1_ISIP .- y0_ISIP, u1 .- u0))
P_ISIP = compute_p(ISIPren_ps)
lhs_ISIP = mat_norm2(P_ISIP, x0n_ISIP .- x1n_ISIP) - mat_norm2(P_ISIP, x0 .- x1)

# ISOP case
rhs_ISOP = mat_norm2(M_ISOP, vcat(y1_ISOP .- y0_ISOP, u1 .- u0))
P_ISOP = compute_p(ISOPren_ps)
lhs_ISOP = mat_norm2(P_ISOP, x0n_ISOP .- x1n_ISOP) - mat_norm2(P_ISOP, x0 .- x1)

@test all(lhs_ISIP .<= rhs_ISIP)
@test all(lhs_ISOP .<= rhs_ISOP)