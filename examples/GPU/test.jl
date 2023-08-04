cd(@__DIR__)
using Pkg
Pkg.activate("../")

using BenchmarkTools
using CUDA
using Flux
using RobustNeuralNetworks

device = gpu
T = Float32

# Model sizes
nu, nx, nv, ny = 4, 5, 10, 2

# Build model
ren_ps = LipschitzRENParams{T}(nu, nx, nv, ny, 1; nl=relu)
ren = REN(ren_ps) |> device
# ren = DiffREN(ren_ps) #|> device

# TODO: If we use DiffREN, construction of REN is slow and better suited to CPU. Think of a good way around this. It also means having model = REN(ren_ps) |> gpu in the loss function. Not ideal...

# Data
batches = 1000000
u = rand(T, nu, batches)      |> device
x = init_states(ren, batches) |> device

function to_dev(ren, x, u, device)
    r = ren |> device
    x1 = x |> device
    u1 = u |> device
    return r, x1, u1
end

# Time on the CPU
println("Calling REN on CPU with $batches batches")
ren, x, u = to_dev(ren, x, u, cpu)
@btime x1, yr = ren(x, u);

# Time on GPU
println("Calling REN on GPU with $batches batches")
ren, x, u = to_dev(ren, x, u, gpu)
@btime x1, yr = ren(x, u);

println()