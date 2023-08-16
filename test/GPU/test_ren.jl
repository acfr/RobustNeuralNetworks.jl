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

function test_ren_device(device, construct, args...; nu=4, nx=5, nv=10, ny=4, 
                         nl=tanh, batches=4, tmax=3, is_diff=false, T=Float32,
                         do_time=true)

    # Build the ren
    model = construct{T}(nu, nx, nv, ny, args...; nl, rng) |> device
    is_diff && (model = DiffREN(model))

    # Create dummy data
    us = [randn(rng, T, nu, batches) for _ in 1:tmax] |> device
    ys = [randn(rng, T, ny, batches) for _ in 1:tmax] |> device
    x0 = init_states(model, batches) |> device

    # Dummy loss function
    function loss(model, x, us, ys)
        m = is_diff ? model : REN(model)
        J = 0
        for t in 1:tmax
            x, y = m(x, us[t])
            J += Flux.mse(y, ys[t])
        end
        return J
    end

    # Run and time, running it once first to check it works
    print("Forwards: ")
    l = loss(model, x0, us, ys)
    do_time && (@btime $loss($model, $x0, $us, $ys))

    print("Reverse:  ")
    g = gradient(loss, model, x0, us, ys)
    do_time && (@btime $gradient($loss, $model, $x0, $us, $ys))

    return l, g
end

# Test all types and combinations
γ = 10
ν = 10

nu, nx, nv, ny = 4, 5, 10, 4
X = randn(rng, ny, ny)
Y = randn(rng, nu, nu)
S = randn(rng, nu, ny)

Q = -X'*X
R = S * (Q \ S') + Y'*Y

function test_rens(device)

    d = device === cpu ? "CPU" : "GPU"
    println("\nTesting RENs on ", d, ":")
    println("--------------------\n")

    println("Contracting REN:\n")
    test_ren_device(device, ContractingRENParams)
    println("\nContracting DiffREN:\n")
    test_ren_device(device, ContractingRENParams; is_diff=true)

    println("\nPassive REN:\n")
    test_ren_device(device, PassiveRENParams, ν)
    println("\nPassive DiffREN:\n")
    test_ren_device(device, PassiveRENParams, ν; is_diff=true)

    println("\nLipschitz REN:\n")
    test_ren_device(device, LipschitzRENParams, γ)
    println("\nLipschitz DiffREN:\n")
    test_ren_device(device, LipschitzRENParams, γ; is_diff=true)

    println("\nGeneral REN:\n")
    test_ren_device(device, GeneralRENParams, Q, S, R)
    println("\nGeneral DiffREN:\n")
    test_ren_device(device, GeneralRENParams, Q, S, R; is_diff=true)

    return nothing
end

test_rens(cpu)
test_rens(gpu)
