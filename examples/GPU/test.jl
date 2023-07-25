cd(@__DIR__)
using Pkg
Pkg.activate("../")

using CUDA
using Flux
using RobustNeuralNetworks

# Standard with Flux
m = Chain(Flux.Dense(10 => 5, Flux.relu), Flux.Dense(5 => 4, Flux.relu)) |> gpu
x = rand(10) |> gpu
m(x)

# Test with LBDN
lbdn = DiffLBDN(DenseLBDNParams{Float32}(10, [5], 4)) |> gpu
lbdn(x)