# This file is a part of RobustNeuralNetworks.jl. License is MIT: https://github.com/acfr/RobustNeuralNetworks.jl/blob/main/LICENSE 

cd(@__DIR__)
using Pkg
Pkg.activate("../")

using CairoMakie
using Flux
using Printf
using Random
using RobustNeuralNetworks

# Random seed for consistency
rng = Xoshiro(0)

# Function to estimate
f(x) = x < 0 ? 0 : 1

# Training data
dx = 0.01
xs = -0.3:dx:0.3
ys = f.(xs)
data = zip(xs,ys)

# Model specification
nu = 1                  # Number of inputs
ny = 1                  # Number of outputs
nh = fill(16,4)         # 4 hidden layers, each with 16 neurons
γ = 10                  # Lipschitz bound of 10

# Set up model: define parameters, then create model
model_ps = DenseLBDNParams{Float64}(nu, nh, ny, γ; rng)
model = DiffLBDN(model_ps)

# Loss function
loss(model,x,y) = Flux.mse(model([x]),[y]) 

# Check fit error/slope during training
mse(model, xs, ys) = sum(loss.((model,), xs, ys)) / length(xs)
lip(model, xs, dx) = maximum(abs.(diff(model(xs'), dims=2)))/dx

# Callback function to show results while training
function progress(model, iter, xs, ys, dx) 
    fit_error = round(mse(model, xs, ys), digits=4)
    slope = round(lip(model, xs, dx), digits=4)
    @show iter fit_error slope
    println()
end

# Define hyperparameters
num_epochs = 300
lr = 2e-4

# Train with the Adam optimiser
opt_state = Flux.setup(Adam(lr), model)
for i in 1:num_epochs
    Flux.train!(loss, model, data, opt_state)
    (i % 50 == 0) && progress(model, i, xs, ys, dx)
end

# Print out lower-bound on Lipschitz constant
Empirical_Lipschitz = lip(model, xs, dx)
@printf "Empirical lower Lipschitz bound: %.2f\n" Empirical_Lipschitz

# Create a figure
fig = Figure(resolution = (600, 400))
ax = Axis(fig[1,1], xlabel="x", ylabel="y")

get_best(x) = x<-0.05 ? 0 : (x<0.05 ? 10x + 0.5 : 1)
ybest = get_best.(xs)
ŷ = map(x -> model([x])[1], xs)

lines!(xs, ys, label = "Data")
lines!(xs, ybest, label = "Slope restriction = 10.0")
lines!(xs, ŷ, label = "LBDN slope = $(round(Empirical_Lipschitz; digits=2))")
axislegend(ax, position=:lt)
display(fig)
save("../results/lbdn-curvefit/lbdn_curve_fit.svg", fig)
