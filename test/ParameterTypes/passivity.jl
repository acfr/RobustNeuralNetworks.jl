# This file is a part of RobustNeuralNetworks.jl. License is MIT: https://github.com/acfr/RobustNeuralNetworks.jl/blob/main/LICENSE 

using LinearAlgebra
using Random
using RobustNeuralNetworks
using Test

# include("../test_utils.jl")

rng = MersenneTwister(42)

"""
Test incrementally strictly input passivity (isip)
"""
batches = 100
nu, nx, nv, ny = 6, 5, 10, 6
ν = 2.0
ρ = 10.0
T = 100

# Test constructor
isip_ren_ps = PassiveRENParams{Float64}(nu, nx, nv, ny, ν, 0; init=:random, rng)
isip_ren = REN(isip_ren_ps)

# Different inputs with different initial conditions
u0 = 10*randn(rng, nu, batches)
u1 = rand(rng, nu, batches)

x0 = randn(rng, nx, batches)
x1 = randn(rng, nx, batches)

# Simulate
x0n_isip, y0_isip = isip_ren(x0, u0)
x1n_isip, y1_isip = isip_ren(x1, u1)

# Dissipation condition
ν = isip_ren_ps.ν
Q_isip = zeros(ny, ny)
R_isip = -2ν * Matrix(I, nu, nu)
S = Matrix(I, nu, ny)

# Test passivity
M_isip = [Q_isip S'; S R_isip]
P_isip = compute_p(isip_ren_ps)
rhs_isip = mat_norm2(M_isip, vcat(y1_isip .- y0_isip, u1 .- u0))
lhs_isip = mat_norm2(P_isip, x0n_isip .- x1n_isip) - mat_norm2(P_isip, x0 .- x1)

@test all(lhs_isip .<= rhs_isip)

"""
Test incrementally strictly output passivity (isop)
"""
isop_ren_ps = PassiveRENParams{Float64}(nu, nx, nv, ny, 0, ρ; init=:random, rng)
isop_ren = REN(isop_ren_ps)

# Simulate
x0n_isop, y0_isop = isop_ren(x0, u0)
x1n_isop, y1_isop = isop_ren(x1, u1)

# Dissipation condition
ρ = isop_ren_ps.ρ
Q_isop = -2ρ * Matrix(I, ny, ny)
R_isop = zeros(nu, nu)
S = Matrix(I, nu, ny)

# Test passivity
M_isop = [Q_isop S'; S R_isop]
P_isop = compute_p(isop_ren_ps)
rhs_isop = mat_norm2(M_isop, vcat(y1_isop .- y0_isop, u1 .- u0))
lhs_isop = mat_norm2(P_isop, x0n_isop .- x1n_isop) - mat_norm2(P_isop, x0 .- x1)

@test all(lhs_isop .<= rhs_isop)
