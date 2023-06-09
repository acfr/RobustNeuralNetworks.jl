# Image Classification with LBDN

Our next example features an LBDN trained to classify the [MNIST](https://en.wikipedia.org/wiki/MNIST_database) dataset. We showed in [Wang & Manchester (2023)](https://doi.org/10.48550/arXiv.2301.11526) that training image classifiers with LBDNs makes them robust to adversarial attacks thanks to the built-in Lipschitz bound. In this example, we will demonstrate how to train an LBDN model on the MNIST dataset with the following steps:
1. Load the training and test data
2. Define a Lipschitz-bounded model
3. Define a loss function
4. Train the model to minimise the loss function
5. Evaluate the trained model
6. Investigate robustness

For details on how Lipschitz bounds increase classification robustness and reliability, please see the [paper](https://doi.org/10.48550/arXiv.2301.11526).

## 1. Load the data

Let's start by loading the training and test data. [`MLDatasets.jl`](https://juliaml.github.io/MLDatasets.jl/stable/) contains a number of common machine-learning datasets, including the [MNIST dataset](https://juliaml.github.io/MLDatasets.jl/stable/datasets/vision/#MLDatasets.MNIST). To load the full dataset of 60,000 training images and 10,000 test images, one would run the following code.

```julia
using MLDatasets: MNIST

# Get MNIST training and test data
T = Float32
x_train, y_train = MNIST(T, split=:train)[:]
x_test,  y_test  = MNIST(T, split=:test)[:]
```

For the purposes of this example, we'll load a small subset of the dataset.
```@example mnist
using BSON

data = BSON.load("../../src/assets/lbdn-mnist/mnist_data.bson")
x_train, y_train = data["x_train"], data["y_train"]
x_test,  y_test  = data["x_test"],  data["y_test"]

println("Train/test features: ", size(x_train), " / ", size(x_test))
println("Train/test labels:   ", size(y_train), " / ", size(y_test))
```

The feature matrices `x_train` and `x_test` are three-dimensional arrays where each 28x28 layer contains pixel data for a single handwritten number from 0 to 9. The labels `y_train` and `y_test` are vectors containing the classification of each image as a number from 0 to 9. We can convert each of these to an input/output format better suited to training with [`Flux.jl`](https://fluxml.ai/).

```@example mnist
using Flux
using Flux: OneHotMatrix

# Reshape features for model input
x_train = Flux.flatten(x_train)
x_test  = Flux.flatten(x_test)

# Encode categorical outputs and store training data
y_train = Flux.onehotbatch(y_train, 0:9)
y_test  = Flux.onehotbatch(y_test,  0:9)
train_data = [(x_train, y_train)]

println("Features: ", size(x_test))
println("Labels:   ", size(y_test))
```

Features are now stored in a `Matrix` where each column contains pixel data from a single image, and the labels have been converted to a `OneHotMatrix` where each column contains a 1 in the row corresponding to the image's classification (eg: row 3 for an image showing the number 2).


## 2. Define a model

We can now construct an LBDN model to train on the MNIST dataset. The larger the model, the better the classification accuracy will be, at the cost of longer training time. For example, using an LBDN with three hidden layers of 128 neurons each would achieve a classification accuracy of approximately 99% on the full MNIST dataset. For this example, we'll use a smaller network and set a Lipschitz bound of `γ = 5.0` just to demonstrate the method.

```@example mnist
using Random
using RobustNeuralNetworks

# Random seed for consistency
rng = MersenneTwister(42)

# Model specification
nu = 28*28              # Number of inputs (size of image)
ny = 10                 # Number of outputs (possible classifications)
nh = fill(64,2)         # 2 hidden layers, each with 64 neurons
γ  = 5                  # Lipschitz bound of 5.0

# Set up model: define parameters, then create model
T = Float32
model_ps = DenseLBDNParams{T}(nu, nh, ny, γ; rng)
model = Chain(DiffLBDN(model_ps), Flux.softmax)

println(typeof(model))
```

The `model` contains a callable [`DiffLBDN`](@ref) model constructed from its direct parameterisation, which is defined by an instance of [`DenseLBDNParams`](@ref) (see the [Package Overview](@ref) for more detail). The output is converted to a probability distribution using a [`softmax`](https://fluxml.ai/Flux.jl/stable/models/nnlib/#NNlib.softmax) layer. Note that all [`AbstractLBDN`](@ref) models can be combined with traditional neural network layers using [`Flux.Chain`](https://fluxml.ai/Flux.jl/stable/models/layers/#Flux.Chain). An alternative approach would be to use [`SandwichFC`](@ref) layers to build the network.


## 3. Define a loss function

A typical loss function for training on datasets with discrete labels is the cross entropy loss. We can use the [`crossentropy`](https://fluxml.ai/Flux.jl/stable/models/losses/#Flux.Losses.crossentropy) loss function shipped with `Flux.jl`.

```@example mnist
# Loss function
loss(model,x,y) = Flux.crossentropy(model(x), y)
```


## 4. Train the model

Before training the model to minimise the cross entropy loss, we can set up a callback function to evaluate the model performance during training.

```@example mnist
using Statistics

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
```

Let's train the model over 600 epochs using two learning rates: `1e-3` for the first 300, and `1e-4` for the last 300. We'll use the [`Adam`](https://fluxml.ai/Flux.jl/stable/training/optimisers/#Flux.Optimise.Adam) optimiser and the default [`Flux.train!`](https://fluxml.ai/Flux.jl/stable/training/reference/#Flux.Optimise.train!-NTuple{4,%20Any}) method. Once the model has been trained, we can save it for later with the [`BSON`](https://github.com/JuliaIO/BSON.jl) package.

```julia
using BSON

# Train with the Adam optimiser, and display progress every 50 steps
function train_mnist!(model, data; num_epochs=300, lrs=[1e-3,1e-4])

    opt_state = Flux.setup(Adam(lrs[1]), model)
    for k in eachindex(lrs)
        opt_state = Flux.setup(Adam(lrs[k]), model)
        for i in 1:num_epochs
            Flux.train!(loss, model, data, opt_state)
            (i % 50 == 0) && progress(model, i)
        end
        (k != length(lrs)) && Flux.adjust!(opt_state, lrs[k+1])
    end
end

# Save the model for later
bson("lbdn_mnist.bson", Dict("model" => model))
```

Running the training loop can take a few minutes, so here's one we prepared earlier. The model was trained on the full MNIST dataset (60,000 training images, 10,000 test images).

```@example mnist
model = BSON.load("../../src/assets/lbdn-mnist/lbdn_mnist.bson")["model"]
println(typeof(model))
```

## 5. Evaluate the trained model

Our final model has a test accuracy of about 98% on this small subset of the MNIST dataset. For those interested, it achieves 97.6% accuracy on the full 10,000-image test set. We could improve this further by (for example) using a larger model, training the model for longer, or fine-tuning the learning rate. 

```@example mnist
# Print final results
train_acc = accuracy(model, x_train, y_train)*100
test_acc  = accuracy(model, x_test,  y_test)*100
println("Training accuracy: $(round(train_acc,digits=2))%")
println("Test accuracy:     $(round(test_acc,digits=2))%")
```

Let's have a look at some examples too.
```@example mnist
using CairoMakie

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
save("lbdn_mnist.svg", f1)
```
![](lbdn_mnist.svg)


## 6. Investigate robustness

The main advantage of using an LBDN for image classification is its built-in robustness to noise (or attacks) added to the image data. This robustness is given by the Lipschitz bound. As explained in the [Package Overview](@ref), a Lipschitz bound effectively defines how "smooth" the network is: the smaller the Lipschitz bound, the less the network outputs will change as the inputs vary. For example, small amounts of noise added to the image will be less likely to change its classification. A detailed investigation into this effect is presented in [Wang & Manchester (2023)](https://doi.org/10.48550/arXiv.2301.11526).

We can see this effect first-hand by comparing the LBDN to a standard MLP built from `Flux.Dense` layers. Let's first create a `dense` network with the same layer structure as the LBDN.
```@example mnist
init = Flux.glorot_normal(rng)
initb(n) = Flux.glorot_normal(rng, n)
dense = Chain(
    Dense(nu, nh[1], Flux.relu; init, bias=initb(nh[1])),
    Dense(nh[1], nh[2], Flux.relu; init, bias=initb(nh[2])),
    Dense(nh[2], ny; init, bias=initb(ny)),
    Flux.softmax
)
```

We can train it with the same `train_mnist!()` function from earlier, which takes a few minutes to run. Here's one we prepared earlier, with some statistics on its performance. On the full data set, it achieves a test accuracy of 96.9%.
```@example mnist
# Load the model
dense = BSON.load("../../src/assets/lbdn-mnist/dense_mnist.bson")["model"]

# Print final results
train_acc = accuracy(dense, x_train, y_train)*100
test_acc  = accuracy(dense, x_test,  y_test)*100
println("Training accuracy: $(round(train_acc,digits=2))%")
println("Test accuracy:     $(round(test_acc,digits=2))%")
```

To test robustness, we'll add uniformly-sampled random noise in the range ``[-\epsilon, \epsilon]`` to the test data for a range of noise magnitudes ``\epsilon \in [0, 200/255].`` We can record the test accuracy for each perturbation size and store it for plotting.
```@example mnist
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
lines!(ax1, ϵs, lbdn_error, label="LBDN (γ=5)")
lines!(ax1, ϵs, dense_error, label="Dense")

xlims!(ax1, 0, 0.8)
axislegend(ax1, position=:lb)
save("lbdn_mnist_robust.svg", fig)
```
![](lbdn_mnist_robust.svg)

Plotting the results very clearly shows that the `dense` network, which has no guarantees on its Lipschitz bound, quickly loses its accuracy as small amounts of noise are added to the image. On the other hand, the LBDN `model` maintains its accuracy even when the (maximum) perturbation size is as much as 80% of the maximum pixel values. Image classification is one of the most promising use-cases of LBDN models, and this is exactly why.