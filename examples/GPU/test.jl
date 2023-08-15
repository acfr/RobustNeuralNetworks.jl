cd(@__DIR__)
using Pkg
Pkg.activate("../")

using BenchmarkTools
using CUDA
using Flux
using Random
using RobustNeuralNetworks

T = Float32
rng = Xoshiro(42)

# Model parameters
nu, ny, γ = 2, 3, 1
nh = [10,5] # TODO: Switch this to a tuple, it really should be immutable

# Build model
model_ps = DenseLBDNParams{T}(nu, nh, ny, γ; rng) #, learn_γ=true)
# model = LBDN(model_ps)
model = DiffLBDN(model_ps)

# Data
batches = 1000
us = randn(rng, T, nu, batches)
ys = randn(rng, T, ny, batches)

function to_dev(lbdn, u, device)
    m = lbdn |> device
    u1 = u |> device
    return m, u1
end

# Time on the CPU
println("Calling LBDN on CPU with $batches batches")
lbdn, u = to_dev(model, us, cpu)
@btime y1 = lbdn(u);

# Time on GPU
println("Calling LBDN on GPU with $batches batches")
lbdn, u = to_dev(model, us, gpu)
@btime y1 = lbdn(u);

# TODO: Remove scalar array indexing

println()
