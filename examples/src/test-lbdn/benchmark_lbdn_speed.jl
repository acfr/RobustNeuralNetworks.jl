cd(@__DIR__)
using Pkg
Pkg.activate("../../")

using CairoMakie
using BenchmarkTools
using Flux
using Random
using RobustNeuralNetworks
using Zygote: pullback

Random.seed!(0)

# Set up model
nu, ny, γ = 1, 1, 1
# nh        = [10,5,5,15]
nh        = fill(10,2)
model_ps  = DenseLBDNParams{Float64}(nu, nh, ny, γ)
model     = DiffLBDN(model_ps)
ps        = Flux.params(model)

# Function to estimate
f(x) = sin(x)+(1/N)*sin(N*x)
     
# Training data
N  = 5
dx = 0.1
xs = 0:dx:2π
ys = f.(xs)
T  = length(xs)
data = zip(xs,ys)

# Training details
lr = 1e-4
opt = ADAM(lr)
loss(x::AbstractMatrix, y::AbstractMatrix) = Flux.mse(model(x),y)

# Test forward pass
u = rand(nu,10)
@btime model(u);

# Test back propagation
function batch_update()
    J, back = pullback(() -> loss(xs', ys'), ps)
    ∇J = back(one(J)) 
    Flux.update!(opt, ps, ∇J)
end
@btime batch_update();

# Make a figure of timing
n = [1,2,3,4,5,8]

forward_t = [12.968, 24.027, 33.998, 44.317, 55.279, 88.647]
forward_a = [81, 117, 153, 189, 225, 333]

backward_t = [222.790, 311.988, 388.723, 472.197, 568.571, 806.951]
backward_a = [993, 1325, 1656, 1987, 2319, 3356]

f1 = Figure()
ax = Axis(f1[1,1], xlabel="Number of layers, 10 hidden nodes each", ylabel="Time (μs)")
lines!(ax, n, forward_t, label="Forward")
lines!(ax, n, backward_t, label="Backward")
axislegend(ax)
display(f1)

f2 = Figure()
ax = Axis(f2[1,1], xlabel="Number of layers, 10 hidden nodes each", ylabel="Allocations")
lines!(ax, n, forward_a, label="Forward")
lines!(ax, n, backward_a, label="Backward")
axislegend(ax)
display(f2)