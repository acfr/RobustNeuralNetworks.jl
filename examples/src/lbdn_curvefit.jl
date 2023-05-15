cd(@__DIR__)
using Pkg
Pkg.activate("../")

using CairoMakie
using Flux
using Random
using RobustNeuralNetworks

# Random seed for consistency
rng = MersenneTwister(42)

# Model specification
nu = 1                  # Number of inputs
ny = 1                  # Number of outputs
nh = fill(15,4)         # 4 hidden layers, each with 15 neurons
γ = 1                   # Lipschitz bound of 1

# Set up model: define parameters, then create model
model_ps = DenseLBDNParams{Float64}(nu, nh, ny, γ; rng=rng)
model = DiffLBDN(model_ps)

# Function to estimate
N = 5
f(x) = sin(x)+(1/N)*sin(N*x)

# Training data
dx = 0.1
xs = 0:dx:2π
ys = f.(xs)
data = zip(xs,ys)

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
num_epochs = [400, 200]
lrs = [2e-4, 5e-5]

# Train with the Adam optimiser
for k in eachindex(lrs)
    opt_state = Flux.setup(Adam(lrs[k]), model)
    for i in 1:num_epochs[k]
        Flux.train!(loss, model, data, opt_state)
        (i % 50 == 0) && progress(model, i, xs, ys, dx)
    end
end

# Create a figure
f1 = Figure(resolution = (600, 400))
ax = Axis(f1[1,1], xlabel="x", ylabel="y")

ŷ = map(x -> model([x])[1], xs)
lines!(xs, ys, label = "Data")
lines!(xs, ŷ, label = "LBDN")
axislegend(ax)
display(f1)
save("../results/lbdn_curve_fit.svg", f1)

# Print out lower-bound on Lipschitz constant
Empirical_Lipschitz = lip(model, xs, dx)
println("Empirical lower Lipschitz bound: ", round(Empirical_Lipschitz; digits=2))
