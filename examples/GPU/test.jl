cd(@__DIR__)
using Pkg
Pkg.activate("../")

using CUDA
using Flux
using RobustNeuralNetworks

T = Float32
device = gpu

# Standard with Flux
m = Chain(Flux.Dense(10 => 5, Flux.relu), Flux.Dense(5 => 4, Flux.relu)) |> device
u = rand(T, 10) |> device
m(u)

# Test with LBDN
nu, nx, nv, ny = 4, 5, 10, 2
ren_ps = ContractingRENParams{T}(nu, nx, nv, ny)
model = DiffREN(ren_ps)
x = init_states(model) |> device
ren = model |> gpu

# TESTED: removed all gpu stuff from repo, and this sent elements to the GPU. Errors calling the model though... will need to figure that one out later.