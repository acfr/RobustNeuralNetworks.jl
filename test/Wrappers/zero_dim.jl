# This file is a part of RobustNeuralNetworks.jl. License is MIT: https://github.com/acfr/RobustNeuralNetworks.jl/blob/main/LICENSE 

using Flux
using Random
using RobustNeuralNetworks
using Test

function (m::AbstractREN{T})(
    xt::AbstractVecOrMat, 
    ut::AbstractVecOrMat,
    explicit::ExplicitRENParams{T}
) where T

    # Allocate bias vectors to avoid error when nv = 0 or nx = 0
    bv = (m.nv == 0) ? 0 : explicit.bv
    bx = (m.nx == 0) ? 0 : explicit.bx

    b = explicit.C1 * xt + explicit.D12 * ut .+ bv
    wt = RobustNeuralNetworks.tril_eq_layer(m.nl, explicit.D11, b)
    xt1 = explicit.A * xt + explicit.B1 * wt + explicit.B2 * ut .+ bx
    yt = explicit.C2 * xt + explicit.D21 * wt + explicit.D22 * ut .+ explicit.by

    return xt1, yt
end

"""
Test that backpropagation runs and parameters change
"""
batches = 10
nu, nx, nv, ny = 4, 0, 0, 2
γ = 10
model_ps = LipschitzRENParams{Float64}(nu, nx, nv, ny, γ)

# Dummy data
us = randn(nu, batches)
ys = randn(ny, batches)
data = [(us, ys)]

# Dummy loss function just for testing
function loss(model_ps, u, y)
    m = REN(model_ps)
    x0 = init_states(m, size(u,2))
    x1, y1 = m(x0, u)
    return Flux.mse(y1, y) + sum(x1.^2)
end

# Test it
model = REN(model_ps);
x0 = init_states(model, batches)

# @btime model(x0, us)
# @btime Flux.gradient(loss, model_ps, us, ys)
# println()

# Make sure batch update actually runs
opt_state = Flux.setup(Adam(0.01), model)
gs = Flux.gradient(loss, model, us, ys)
Flux.update!(opt_state, model, gs[1])
@test !isempty(gs)
