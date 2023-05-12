using Flux
using Random
using RobustNeuralNetworks
using Test

include("../test_utils.jl")

"""
Test that backpropagation runs and parameters change
"""
batches = 10
nu, ny, γ = 2, 3, 1
nh = [10,5]
model_ps = DenseLBDNParams{Float64}(nu, nh, ny, γ)
model = DiffLBDN(model_ps)

# Dummy data
us = randn(nu, batches)
ys = randn(ny, batches)
data = [(us[:,k], ys[:,k]) for k in 1:batches]

# Dummy loss function just for testing
loss(m, u, y) = Flux.mse(m(u), y)

# Check if parameters change after a Flux update
ps1 = deepcopy(Flux.params(model))
opt_state = Flux.setup(Adam(0.01), model)
Flux.train!(loss, model, data, opt_state)
ps2 = Flux.params(model)

@test !any(ps1 .≈ ps2)
