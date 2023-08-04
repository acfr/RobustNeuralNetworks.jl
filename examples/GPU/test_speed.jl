# This file is a part of RobustNeuralNetworks.jl. License is MIT: https://github.com/acfr/RobustNeuralNetworks.jl/blob/main/LICENSE 
cd(@__DIR__)
using Pkg
Pkg.activate("../")

using BenchmarkTools
using Flux
using Random
using RobustNeuralNetworks

rng = Xoshiro(42)

function test_diffren_speed(device, construct, args...; nu=4, nx=5, nv=10, ny=2, 
                            nl=relu, batches=100, tmax=5, isdiff=true, T=Float32)

    # Build the ren
    model = construct{T}(nu, nx, nv, ny, args...; nl, rng)
    is_diff && (model = DiffREN(model))

    # Create dummy data
    us = [randn(rng, T, nu, batches) for _ in 1:tmax] |> device
    ys = [randn(rng, T, ny, batches) for _ in 1:tmax] |> device
    x0 = init_states(model, batches) |> device

    # Dummy loss function
    function loss(model, x, us, ys)
        J = 0
        for t in 1:tmax
            x, y = model(x, us[t])
            J += Flux.mse(y, ys[t])
        end
        return J
    end

    # Run it once to check it works
    l = loss(model, x0, us, ys)
    g = gradient(loss, model, x0, us, ys)

    # Time it
    print("Forwards: ")
    @btime $loss($model, $x0, $us, $ys)
    print("Reverse:  ")
    @btime $gradient($loss, $model, $x0, $us, $ys)

    return l, g
end

l, g = test_diffren_speed(
    cpu,
    ContractingRENParams;
    batches=100,
    tmax=5,
)
println()

# TODO: Write function for non-diff REN too!