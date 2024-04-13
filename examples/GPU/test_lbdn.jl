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

function test_lbdn_device(device; nu=2, nh=[10, 5], ny=4, γ=10, nl=tanh, 
                          batches=4, is_diff=false, do_time=true, T=Float32)

    # Build model
    model = DenseLBDNParams{T}(nu, nh, ny, γ; nl, rng) |> device
    is_diff && (model = DiffLBDN(model))

    # Create dummy data
    us = randn(rng, T, nu, batches) |> device
    ys = randn(rng, T, ny, batches) |> device

    # Dummy loss function
    function loss(model, u, y)
        m = is_diff ? model : LBDN(model)
        return Flux.mse(m(u), y)
    end

    # Run and time, running it once to check it works
    print("Forwards: ")
    l = loss(model, us, ys)
    do_time && (@btime $loss($model, $us, $ys))

    print("Reverse:  ")
    g = gradient(loss, model, us, ys)
    do_time && (@btime $gradient($loss, $model, $us, $ys))

    return l, g
end

function test_lbdns(device)

    d = device === cpu ? "CPU" : "GPU"
    println("\nTesting LBDNs on ", d, ":")
    println("--------------------\n")

    println("Dense LBDN:\n")
    test_lbdn_device(device)
    println("\nDense DiffLBDN:\n")
    test_lbdn_device(device; is_diff=true)

    return nothing
end

test_lbdns(cpu)
test_lbdns(gpu)
