# This file is a part of RobustNeuralNetworks.jl. License is MIT: https://github.com/acfr/RobustNeuralNetworks.jl/blob/main/LICENSE 

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
rng = MersenneTwister(42)

# Model specification
nu = 28*28              # Number of inputs (size of image)
ny = 10                 # Number of outputs (possible classifications)
nh = fill(64,2)         # 2 hidden layers, each with 64 neurons
γ  = 5                  # Lipschitz bound of 5

# Set up model: define parameters, then create model
T = Float32
model_ps = DenseLBDNParams{T}(nu, nh, ny, γ; rng)
model = Chain(DiffLBDN(model_ps), Flux.softmax)

# Get MNIST training and test data
x_train, y_train = MNIST(T, split=:train)[:]
x_test,  y_test  = MNIST(T, split=:test)[:]

# Reshape features for model input
x_train = Flux.flatten(x_train)
x_test  = Flux.flatten(x_test)

# Encode categorical outputs and store data
y_train = Flux.onehotbatch(y_train, 0:9)
y_test  = Flux.onehotbatch(y_test,  0:9)
train_data = [(x_train, y_train)]

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

# Train the model with the ADAM optimiser
function train_mnist!(model, data; num_epochs=300, lrs=[1e-3,1e-4])
    opt_state = Flux.setup(Adam(lrs[1]), model)
    for k in eachindex(lrs)    
        for i in 1:num_epochs
            Flux.train!(loss, model, data, opt_state)
            (i % 50 == 0) && progress(model, i)
        end
        (k < length(lrs)) && Flux.adjust!(opt_state, lrs[k+1])
    end
end

# Train and save the model for later use
train_mnist!(model, train_data)
bson("../results/lbdn-mnist/lbdn_mnist.bson", Dict("model" => model))
model = BSON.load("../results/lbdn-mnist/lbdn_mnist.bson")["model"]

# Print final results
train_acc = accuracy(model, x_train, y_train)*100
test_acc  = accuracy(model, x_test,  y_test)*100
println("LBDN Results: ")
println("Training accuracy: $(round(train_acc,digits=2))%")
println("Test accuracy:     $(round(test_acc,digits=2))%\n")

# Make a couple of example plots
indx = rand(rng, 1:100, 3)
fig = Figure(resolution = (800, 300), fontsize=21)
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
        fig[1,i], xmat, axis=(
            yreversed = true, 
            aspect = DataAspect(), 
            title = "Label: $(yval), Prediction: $(ŷval)",
        )
    )

    # Format the plot
    ax.xticksvisible = false
    ax.yticksvisible = false
    ax.xticklabelsvisible = false
    ax.yticklabelsvisible = false

end
display(fig)
save("../results/lbdn-mnist/lbdn_mnist.svg", fig)


#######################################################################
# Compare robustness to Dense network

# Create a Dense network 
init = Flux.glorot_normal(rng)
initb(n) = Flux.glorot_normal(rng, n)
dense = Chain(
    Dense(nu, nh[1], Flux.relu; init, bias=initb(nh[1])),
    Dense(nh[1], nh[2], Flux.relu; init, bias=initb(nh[2])),
    Dense(nh[2], ny; init, bias=initb(ny)),
    Flux.softmax
)

# Train it and save for later
train_mnist!(dense, train_data)
bson("../results/lbdn-mnist/dense_mnist.bson", Dict("model" => dense))
dense = BSON.load("../results/lbdn-mnist/dense_mnist.bson")["model"]

# Print final results
train_acc = accuracy(dense, x_train, y_train)*100
test_acc  = accuracy(dense, x_test,  y_test)*100
println("Dense results:")
println("Training accuracy: $(round(train_acc,digits=2))%")
println("Test accuracy:     $(round(test_acc,digits=2))%")

# Get test accuracy as we add noise
uniform(x) = 2*rand(rng, T, size(x)...) .- 1
function noisy_test_error(model, ϵ=0)
    noisy_xtest = x_test .+ ϵ*uniform(x_test)
    accuracy(model, noisy_xtest,  y_test)*100
end

ϵs = T.(LinRange(0, 200, 10)) ./ 255
lbdn_error = noisy_test_error.((model,), ϵs)
dense_error = noisy_test_error.((dense,), ϵs)

# Plot results
fig = Figure(resolution=(500,300))
ax1 = Axis(fig[1,1], xlabel="Perturbation size", ylabel="Test accuracy (%)")
lines!(ax1, ϵs, lbdn_error, label="LBDN γ=5")
lines!(ax1, ϵs, dense_error, label="Dense")

xlims!(ax1, 0, 0.8)
axislegend(ax1, position=:lb)
display(fig)
save("../results/lbdn-mnist/lbdn_mnist_robust.svg", fig)
