# This file is a part of RobustNeuralNetworks.jl. License is MIT: https://github.com/acfr/RobustNeuralNetworks.jl/blob/main/LICENSE 

using Flux
using Random
using RobustNeuralNetworks
using Test

# include("../test_utils.jl")

"""
Test that backpropagation runs and parameters change
"""
batches = 10
nu, nx, nv, ny = 1, 10, 0, 1
γ = 10
model_ps = LipschitzRENParams{Float32}(nu, nx, nv, ny, γ)

# Dummy data
us = randn(nu, batches)
ys = randn(ny, batches)
data = [(us[:,k], ys[:,k]) for k in 1:batches]

# Dummy loss function just for testing
function loss(model_ps, u, y)
    m = DiffREN(model_ps)
    x0 = init_states(m, size(u,2))
    x1, y1 = m(x0, u)
    return Flux.mse(y1, y) + sum(x1.^2)
end

# Check if parameters change after a Flux update
ps1 = deepcopy(Flux.params(model_ps))
opt_state = Flux.setup(Adam(0.01), model_ps)
Flux.train!(loss, model_ps, data, opt_state)
ps2 = Flux.params(model_ps)

@test !any(ps1 .≈ ps2)
