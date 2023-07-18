# This file is a part of RobustNeuralNetworks.jl. License is MIT: https://github.com/acfr/RobustNeuralNetworks.jl/blob/main/LICENSE 

using Flux
using Random
using RobustNeuralNetworks
using Test

rng = MersenneTwister(42)

"""
Test that backpropagation runs when nx = 0 and nv = 0
"""
batches = 10
nu, nx, nv, ny = 4, 0, 0, 2
γ = 10
model_ps = LipschitzRENParams{Float64}(nu, nx, nv, ny, γ; rng)

# Dummy data
us = randn(rng, nu, batches)
ys = randn(rng, ny, batches)
data = [(us, ys)]

# Dummy loss function just for testing
function loss(model_ps, u, y)
    m = REN(model_ps)
    x0 = init_states(m, size(u,2))
    x1, y1 = m(x0, u)
    return Flux.mse(y1, y) + sum(x1.^2)
end

# Make sure batch update actually runs
opt_state = Flux.setup(Adam(0.01), model_ps)
gs = Flux.gradient(loss, model_ps, us, ys)
Flux.update!(opt_state, model_ps, gs[1])
@test !isempty(gs)
