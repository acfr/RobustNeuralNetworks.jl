cd(@__DIR__)
using Pkg
Pkg.activate("../../")

using BenchmarkTools
using Flux
using Random
using RobustNeuralNetworks
using Zygote: pullback

Random.seed!(0)

# Set up model
nu, ny, γ = 1, 1, 1
nh        = [10,5,5,15]
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
