cd(@__DIR__)
using Pkg
Pkg.activate("../")

using MLDatasets: MNIST

# Get MNIST training and test data
T = Float32
x_train, y_train = MNIST(T, split=:train)[:]
x_test,  y_test  = MNIST(T, split=:test)[:]

using Flux
using Flux: OneHotMatrix

# Reshape features for model input
x_train = Flux.flatten(x_train)
x_test  = Flux.flatten(x_test)

# Encode categorical outputs and store
y_train = Flux.onehotbatch(y_train, 0:9)
y_test  = Flux.onehotbatch(y_test,  0:9)
data = [(x_train, y_train)]

using RobustNeuralNetworks

# Model specification
nu = 28*28              # Inputs (size of image)
ny = 10                 # Outputs (classifications)
nh = fill(64,2)         # Hidden layers 
γ  = T(5)               # Lipschitz bound 5.0

# Define parameters,create model
model_ps = DenseLBDNParams{T}(nu, nh, ny, γ)
model = Chain(DiffLBDN(model_ps), Flux.softmax)

# model = Chain(
#     (x) -> (sqrt(γ) * x),
#     SandwichFC(nu => nh[1], relu; T),
#     SandwichFC(nh[1] => nh[2], relu; T),
#     (x) -> (sqrt(γ) * x),
#     SandwichFC(nh[2] => ny; output_layer=true, T),
#     Flux.softmax
# )

loss(model,x,y) = Flux.crossentropy(model(x), y)

# Hyperparameters
epochs = 3
lrs = [1e-3,1e-4]

# Train with the Adam optimiser
for k in eachindex(lrs)
    opt_state = Flux.setup(Adam(lrs[k]), model)
    for i in 1:epochs
        Flux.train!(loss, model, data, opt_state)
    end
end

init = Flux.glorot_normal
initb(n) = Flux.glorot_normal(n)
dense = Chain(
    Dense(nu, nh[1], relu; 
          init, bias=initb(nh[1])),
    Dense(nh[1], nh[2], relu; 
          init, bias=initb(nh[2])),
    Dense(nh[2], ny; init, bias=initb(ny)),
    Flux.softmax
)

for k in eachindex(lrs)
    opt_state = Flux.setup(Adam(lrs[k]), model)
    for i in 1:epochs
        Flux.train!(loss, model, data, opt_state)
    end
end

# Get test accuracy as we add noise
using Statistics

uniform(x) = 2*rand(T, size(x)...) .- 1
compare(y, yh) = 
    maximum(yh, dims=1) .== maximum(y.*yh, dims=1)
accuracy(model, x, y) = mean(compare(y, model(x)))
    
function noisy_test_error(model, ϵ)
    noisy_xtest = x_test .+ ϵ*uniform(x_test)
    accuracy(model, noisy_xtest,  y_test)*100
end

ϵs = T.(LinRange(0, 200, 10)) ./ 255
lbdn_error  = noisy_test_error.((model,), ϵs)
dense_error = noisy_test_error.((dense,), ϵs)