#  Old directory: /test/
# ----------------------------------
cd(@__DIR__)
using Pkg
Pkg.activate("./..")

using BenchmarkTools
using LinearAlgebra
using RecurrentEquilibriumNetworks
# using Distributions
"""
Johnny's random testing script: 
1. Passive ren test Done
2. passivity type
3. Next: try training
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

