# This file is a part of RobustNeuralNetworks.jl. License is MIT: https://github.com/acfr/RobustNeuralNetworks.jl/blob/main/LICENSE 

using Flux
using Random
using RobustNeuralNetworks
using Test

rng = MersenneTwister(42)

"""
Test that backpropagation runs and parameters change
"""
batches = 10
nu, nx, nv, ny = 4, 5, 10, 2
γ = 10
ren_ps = LipschitzRENParams{Float64}(nu, nx, nv, ny, γ; rng)
model = DiffREN(ren_ps)

# Dummy data
us = randn(rng, nu, batches)
ys = randn(rng, ny, batches)
data = [(us[:,k], ys[:,k]) for k in 1:batches]

# Dummy loss function just for testing
function loss(m, u, y)
    x0 = init_states(m, size(u,2))
    x1, y1 = m(x0, u)
    return Flux.mse(y1, y) + sum(x1.^2)
end

# Check if parameters change after a Flux update
ps1 = deepcopy(Flux.params(model))
opt_state = Flux.setup(Adam(0.01), model)
Flux.train!(loss, model, data, opt_state)
ps2 = Flux.params(model)

@test !any(ps1 .≈ ps2)
