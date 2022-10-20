using Random
using RecurrentEquilibriumNetworks
using Test

"""
Test that the contracting REN actually does contract
"""
nsteps = 40
batches = 20
nu, nx, nv, ny = 4, 5, 10, 2

# Init. with :cholesky is slower to converge than :random
contracting_ren_ps = ContractingRENParams{Float64}(nu, nx, nv, ny; init=:cholesky, αbar=0.5)
contracting_ren = REN(contracting_ren_ps)

# Set up states and controls
us = [randn(nu, batches) for _ in 1:nsteps]

xs0 = fill(zeros(nx,batches), nsteps)
xs1 = fill(zeros(nx,batches), nsteps)

xs0[1] = randn(nx, batches);
xs1[1] = randn(nx, batches);

# Simulate
for t in 2:nsteps
    xs0[t], _ = contracting_ren(xs0[t-1], us[t])
    xs1[t], _ = contracting_ren(xs1[t-1], us[t])
end

# Test for contraction
@test maximum(abs.(xs0[end] .- xs1[end])) < 1e-8
