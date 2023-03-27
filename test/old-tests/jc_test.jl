#  Old directory: /test/
# ----------------------------------
cd(@__DIR__)
using Pkg
Pkg.activate("./..")

using BenchmarkTools
using LinearAlgebra
using RobustNeuralNetworks
# using Distributions
"""
Johnny's random testing script: 
# 1. Passive ren test Done
# 2. passivity type
3. Next: try training
4. Recap: what was ν again?
"""

nu = 4
nx = 10
nv = 20
ny = 4
T = 100



# Test constructors
Params = PassiveRENParams{Float64}(nu, nx, nv, ny; init=:random)
ren = REN(Params)

u = [randn(nu, 1) for t in 1:T]



# Training parameters
# general
m.nl, m.nu, m.nx, m.nv, m.ny, direct_ps, m.αbar, m.Q, m.S, m.R
# lpsz
m.nl, m.nu, m.nx, m.nv, m.ny, direct_ps, m.αbar, m.γ
# passive
m.nl, m.nu, m.nx, m.nv, m.ny, direct_ps, m.αbar, m.ν