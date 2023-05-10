cd(@__DIR__)
using Pkg
Pkg.activate("../../")

using CairoMakie
using Flux
using Random
using RobustNeuralNetworks

Random.seed!(0)

# Set up model
nu, ny  = 1, 1
nh = fill(10,4)

b(n_out) = Flux.glorot_normal(n_out)
model = Chain(
    Dense(nu    => nh[1], bias=b(nh[1]), Flux.relu, init=Flux.glorot_normal),
    Dense(nh[1] => nh[2], bias=b(nh[2]), Flux.relu, init=Flux.glorot_normal),
    Dense(nh[2] => nh[3], bias=b(nh[3]), init=Flux.glorot_normal),
    Dense(nh[3] => nh[4], bias=b(nh[4]), Flux.relu, init=Flux.glorot_normal),
    Dense(nh[4] => ny,    bias=b(ny), init=Flux.glorot_normal),
)
ps = Flux.params(model)

# TODO: All of the following...
"""
If the model is the following, setting up an optimiser with the new method doesn't seem to work. Try the following:

    model_ps = DenseLBDNParams{Float64}(nu, nh, ny, γ)
    model    = DiffLBDN(model_ps)
    Flux.setup(Adam(0.01), model)

However, if it's a dense model, it works. Eg, running this:

    model = Dense(nu => ny, bias=b(ny), Flux.relu, init=Flux.glorot_normal)
    Flux.setup(Adam(0.01), model)

It might be because the model is not specified as @functor. Need to be careful with how we do this, only certain fields (the params) are learnable...

Try to figure out what the difference is in setup between Flux.Dense and a DiffLBDN model, see if there's something obvious we're missing. Might be worth making your own model (independent of the repository) and seeing what is required to make it Flux-compatible.
"""


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
num_epochs = [200, 200]
lrs = [1e-3, 1e-4]
for k in eachindex(lrs)
    opt = Adam(lrs[k])
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
