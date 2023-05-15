cd(@__DIR__)
using Pkg
Pkg.activate("../")

using Flux
using Flux: OneHotMatrix
using MLDatasets: MNIST
using Random
using RobustNeuralNetworks
using Statistics

# Random seed for consistency
rng = MersenneTwister(42)

# Model specification
nu = 28*28              # Number of inputs (size of image)
ny = 10                 # Number of outputs (possible classifications)
nh = fill(64,2)         # 2 hidden layers, each with 64 neurons
γ = 10                  # Lipschitz bound of 1

# Set up model: define parameters, then create model
model_ps = DenseLBDNParams{Float64}(nu, nh, ny, γ; rng=rng)
model = Chain(
    DiffLBDN(model_ps), 
    Flux.softmax
)

# Get MNIST training and test data
T = Float64
x_train, y_train = MNIST(T, split=:train)[:]
x_test,  y_test  = MNIST(T, split=:test)[:]

# Reshape features for model input
x_train = Flux.flatten(x_train)
x_test  = Flux.flatten(x_test)

# Encode categorical variables on output
y_train = Flux.onehotbatch(y_train, 0:9)
y_test  = Flux.onehotbatch(y_test,  0:9)

# Loss function
loss(model,x,y) = Flux.crossentropy(model(x), y)

# Check % accuracy during training
compare(y::OneHotMatrix, ŷ) = maximum(ŷ, dims=1) .== maximum(y.*ŷ, dims=1)
accuracy(model, x, y::OneHotMatrix) = mean(compare(y, model(x)))

# Callback function to show results while training
function progress(model, iter)
    train_loss = round(loss(model, x_train, y_train), digits=4)
    test_acc = round(accuracy(model, x_test, y_test), digits=4)
    @show iter train_loss test_acc
    println()
end

# Define hyperparameters
num_epochs = [200, 400]
lrs = [1e-3, 1e-4]
data = [(x_train, y_train)]

# Train with the Adam optimiser
for k in eachindex(lrs)
    opt_state = Flux.setup(Adam(lrs[k]), model)
    for i in 1:num_epochs[k]
        Flux.train!(loss, model, data, opt_state)
        (i % 10 == 0) && progress(model, i)
    end
end

# Print final results
train_acc = accuracy(model, x_train, y_train)
test_acc  = accuracy(model, x_test,  y_test)
println("Training accuracy: $(round(train_acc,digits=4)*100)%")
println("Test accuracy:     $(round(test_acc,digits=4)*100)%")
