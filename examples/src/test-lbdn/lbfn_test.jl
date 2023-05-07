cd(@__DIR__)
using Pkg
Pkg.activate("../../")

using Flux
using Random
using Revise
using RobustNeuralNetworks
using CairoMakie
using Zygote: pullback

includet("./lbfn_OLD.jl")

Random.seed!(0)

# Set up model
nu, ny = 1, 1
nh     = [10,5,5,15]
model  = LBFN{Float64}(nu, nh, ny)
ps     = Flux.params(model)

# Function to estimate
f(x) = sin(x)+(1/N)*sin(N*x)

# Training data
N  = 5
dx = 0.1
xs = 0:dx:2π
ys = f.(xs)
T  = length(xs)
data = zip(xs,ys)

# Loss function
loss(x,y) = Flux.mse(model([x]),[y]) 

# Callback function to show results while training
function evalcb(α) 
    fit_error = sqrt(sum(loss.(xs, ys)) / length(xs))
    slope     = maximum(abs.(diff(model(xs'),dims=2)))/dx
    @show α fit_error slope
    println()
end

# Training loop
num_epochs = [400, 200]
lrs = [2e-4, 5e-5]
for k in eachindex(lrs)
    opt = ADAM(lrs[k])
    for i in 1:num_epochs[k]
        Flux.train!(loss, ps, data, opt)
        (i % 10 == 0) && evalcb(lrs[k])
    end
end

# Create a figure
f1 = Figure()
ax = Axis(f1[1,1], xlabel="x", ylabel="y")

ŷ = map(x -> model([x])[1], xs)
lines!(xs, ys, label = "Data")
lines!(xs, ŷ, label = "LBDN")
axislegend(ax)
display(f1)

# Print out lower-bound on Lipschitz constant
Empirical_Lipschitz = maximum(abs.(diff(model(xs'),dims=2)))/dx
println("Empirical lower Lipschitz bound: ", round(Empirical_Lipschitz; digits=2))