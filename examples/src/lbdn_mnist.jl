cd(@__DIR__)
using Pkg
Pkg.activate("../")

using BSON
using CairoMakie
using Flux
using Flux: OneHotMatrix
using MLDatasets: MNIST
using Random
using RobustNeuralNetworks
using Statistics

# Random seed for consistency
rng = MersenneTwister(24)

# Model specification
nu = 28*28              # Number of inputs (size of image)
ny = 10                 # Number of outputs (possible classifications)
nh = fill(64,2)         # 2 hidden layers, each with 64 neurons
γ  = 5                  # Lipschitz bound of 5

# Set up model: define parameters, then create model
model_ps = DenseLBDNParams{Float64}(nu, nh, ny, γ; rng=rng)
model = Chain(DiffLBDN(model_ps), Flux.softmax)

# Get MNIST training and test data
T = Float64
x_train, y_train = MNIST(T, split=:train)[:]
x_test,  y_test  = MNIST(T, split=:test)[:]

# data = BSON.load("assets/lbdn-mnist/mnist_data.bson")
# x_train, y_train = data["x_train"], data["y_train"]
# x_test, y_test = data["x_test"], data["y_test"]

# Reshape features for model input
x_train = Flux.flatten(x_train)
x_test  = Flux.flatten(x_test)

# Encode categorical variables on output
y_train = Flux.onehotbatch(y_train, 0:9)
y_test  = Flux.onehotbatch(y_test,  0:9)

# Loss function
loss(model,x,y) = Flux.crossentropy(model(x), y)

# Check test accuracy during training
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
num_epochs = 300
lrs = [1e-3, 1e-4]
data = [(x_train, y_train)]

# Train with the Adam optimiser
for k in eachindex(lrs)
    opt_state = Flux.setup(Adam(lrs[k]), model)
    for i in 1:num_epochs
        Flux.train!(loss, model, data, opt_state)
        (i % 50 == 0) && progress(model, i)
    end
end

# Save the model for later use
bson("../results/lbdn_mnist.bson", Dict("model" => model))
# model = BSON.load("../results/lbdn_mnist.bson")["model"]

# Print final results
train_acc = accuracy(model, x_train, y_train)*100
test_acc  = accuracy(model, x_test,  y_test)*100
println("Training accuracy: $(round(train_acc,digits=2))%")
println("Test accuracy:     $(round(test_acc,digits=2))%")

# Make a couple of example plots
indx = rand(rng, 1:100, 3)
f1 = Figure(resolution = (800, 300))
for i in eachindex(indx)

    # Get data and do prediction
    x = x_test[:,indx[i]]
    y = y_test[:,indx[i]]
    ŷ = model(x)

    # Reshape data for plotting
    xmat = reshape(x, 28, 28)
    yval = (0:9)[y][1]
    ŷval = (0:9)[ŷ .== maximum(ŷ)][1]

    # Plot results
    ax, _ = image(
        f1[1,i], xmat, axis=(
            yreversed = true, 
            aspect = DataAspect(), 
            title = "True class: $(yval), Prediction: $(ŷval)"
        )
    )

    # Format the plot
    ax.xticksvisible = false
    ax.yticksvisible = false
    ax.xticklabelsvisible = false
    ax.yticklabelsvisible = false

end
display(f1)
save("../results/lbdn_mnist.svg", f1)
