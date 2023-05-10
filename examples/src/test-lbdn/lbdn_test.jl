cd(@__DIR__)
using Pkg
Pkg.activate("../../")

using Flux
using Random
using RobustNeuralNetworks
using CairoMakie
using Zygote: pullback

Random.seed!(0)

# Set up model
nu, ny   = 1, 1
# nh       = [10,5,5,15]
nh       = fill(90,8)
γ        = 1
model_ps = DenseLBDNParams{Float64}(nu, nh, ny, γ)
ps       = Flux.params(model_ps)

# Function to estimate
f(x) = sin(x)+(1/N)*sin(N*x)

# Training data
N  = 5
dx = 0.12
xs = 0:dx:2π
ys = f.(xs)
T  = length(xs)
data = zip(xs,ys)

# Loss function
function loss(x, y) 
    model = LBDN(model_ps)
    return Flux.mse(model([x]),[y])
end

# Callback function to show results while training
function evalcb(α) 
    model     = LBDN(model_ps)
    fit_error = sqrt(sum(loss.(xs, ys)) / length(xs))
    slope     = maximum(abs.(diff(model(xs'),dims=2)))/dx
    @show α fit_error slope
    println()
end


# Set up training loop
num_epochs = 100
lrs = [2e-3, 8e-4, 5e-4]#, 1e-4, 5e-5]
for k in eachindex(lrs)
    opt = NADAM(lrs[k])
    for i in 1:num_epochs

        # Flux.train!(loss, ps, data, opt)

        for d in data
            J, back = pullback(() -> loss(d[1],d[2]), ps)
            ∇J = back(one(J)) 
            Flux.update!(opt, ps, ∇J)   
        end

        (i % 2 == 0) && evalcb(lrs[k])
    end
end

# Final trained model
model = LBDN(model_ps)

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
