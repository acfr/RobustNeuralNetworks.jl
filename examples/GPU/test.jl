cd(@__DIR__)
using Pkg
Pkg.activate("../")

using CUDA
using Flux
using RobustNeuralNetworks

device = gpu
T = Float32

# Model sizes
nu, nx, nv, ny = 4, 5, 10, 2

# Build models
dense = Chain(
    Dense(nu => nv, relu), 
    Dense(nv => ny, relu)
) |> device

ren_ps = ContractingRENParams{T}(nu, nx, nv, ny; nl=relu)
ren = DiffREN(ren_ps) |> device

# Data
batches = 10
u = rand(T, nu, batches)      |> device
x = init_states(ren, batches) |> device

# Call models
println("Calling dense...")
yd = dense(u)

println("Calling REN...")
x1, yr = ren(x, u)

println("Done!")