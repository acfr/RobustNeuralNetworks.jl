# This file is a part of RobustNeuralNetworks.jl. License is MIT: https://github.com/acfr/RobustNeuralNetworks.jl/blob/main/LICENSE 

cd(@__DIR__)
using Pkg
Pkg.activate("../")

using BenchmarkTools
using CUDA
using Flux
using Random
using RobustNeuralNetworks

rng = Xoshiro(42)

function test_sandwich_device(device; batches=400, do_time=true, T=Float32)

    # Model parameters
    nu = 2
    nh = [10, 5]
    ny = 4
    γ = 10
    nl = tanh

    # Build model
    model = Flux.Chain(
        (x) -> (√γ * x),
        SandwichFC(nu => nh[1], nl; T, rng),
        SandwichFC(nh[1] => nh[2], nl; T, rng),
        (x) -> (√γ * x),
        SandwichFC(nh[2] => ny; output_layer=true, T, rng),
    ) |> device

    # Create dummy data
    us = randn(rng, T, nu, batches) |> device
    ys = randn(rng, T, ny, batches) |> device

    # Dummy loss function
    loss(model, u, y) = Flux.mse(model(u), y)

    # Run and time, running it once to check it works
    print("Forwards: ")
    l = loss(model, us, ys)
    do_time && (@btime $loss($model, $us, $ys))

    print("Reverse:  ")
    g = gradient(loss, model, us, ys)
    do_time && (@btime $gradient($loss, $model, $us, $ys))

    return l, g
end

function test_sandwich(device)

    d = device === cpu ? "CPU" : "GPU"
    println("\nTesting Sandwich on ", d, ":")
    println("--------------------\n")

    test_sandwich_device(device)

    return nothing
end

test_sandwich(cpu)
test_sandwich(gpu)
