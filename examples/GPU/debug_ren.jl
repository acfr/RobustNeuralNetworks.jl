# This file is a part of RobustNeuralNetworks.jl. License is MIT: https://github.com/acfr/RobustNeuralNetworks.jl/blob/main/LICENSE 

cd(@__DIR__)
using Pkg
Pkg.activate("..")

using BSON
using CUDA
using Flux
using Random
using RobustNeuralNetworks
using Statistics

rng = Xoshiro(0)
T = Float32
dev = gpu

# Data
data = BSON.load("failure_outputs.bson")
xn = data["xn"] |> dev
xt = data["xt"] |> dev
inputs = data["inputs"] |> dev

# Model
nx = 2
nv = 20
nu = 2
ny = nx
model_ps = ContractingRENParams{T}(nu, nx, nv, ny; output_map=false, rng)
model = DiffREN(model_ps) |> dev

# Loss function: one step ahead error (average over time)
function loss(model, xn, xt, inputs)
    xpred = model(xt, inputs)[1]
    return mean(sum((xn - xpred).^2; dims=1))
end

# Get gradients
for _ in 1:10

    train_loss, ∇J = Flux.withgradient(loss, model, xn, xt, inputs)

    gs = ∇J[1][:params][:direct]
    # println(train_loss)
    # println(gs)
    # println(gs[:X])
    println(gs[:ρ])
    println(gs[:ϵ])
end

# NOTE: This only produced the NaN result once (now twice)
# I wonder what happens if I have everything inside smaller functions...?
# And does it help to push to |> dev on the original data...?

# There is a direct correlation between the forward pass evaluating differently and the gradients appearing as NaN in that step. Investigate this more.


# Only X, ρ, and ϵ have NaN gradients. For a contracting REN, these are all that's used prior to calling hmatrix_to_explicit(). In terms of back-propagation, perhaps the gradient of hmatrix_to_explicit is NaN and so then the multiplication is too?

# TODO: Write a script that traces the gradients back to the first NaN

# TODO: Also need to sort out the forward pass... this is still not "deterministic"

# function test_me()
#     x0 = model(xt, inputs)[1]
#     all_good = true
#     for _ in 1:10000
#         xpred = model(xt, inputs)[1]
#         if !(xpred ≈ x0)
#             all_good = false
#             println(xpred .- x0)
#         end
#         x0 = xpred
#     end
#     return all_good
# end

# println("Evaluates correctly? ", test_me())