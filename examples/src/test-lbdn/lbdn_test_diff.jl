cd(@__DIR__)
using Pkg
Pkg.activate("../../")

using CairoMakie
using Flux
using Random
using RobustNeuralNetworks
using Zygote: pullback

Random.seed!(0)

# Set up model
nu, ny   = 1, 1
nh       = [10,5,5,15]
γ        = 1
model_ps = DenseLBDNParams{Float64}(nu, nh, ny, γ)
model    = DiffLBDN(model_ps)
ps       = Flux.params(model)

p1 = deepcopy(ps)

u1 = rand(nu,10)
y1 = model(u1)


# Function to estimate
f(x) = sin(x)+(1/N)*sin(N*x)
# f(x) = ((x > -1 && x < 0) || x > 1) ? 1 : 0
     
# Training data
N  = 5
dx = 0.1
xs = 0:dx:2π
# xs = -2:dx:2
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
num_epochs = 200
lrs = [1e-3, 1e-4, 5e-5]
for k in eachindex(lrs)
    opt = ADAM(lrs[k])
    for i in 1:num_epochs
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

p2 = deepcopy(ps)
println(p1 .== p2)
