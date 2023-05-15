# Fitting a Curve with LBDN

For our first example, let's fit a Lipschitz-bounded Deep Network (LBDN) to a curve in one dimension. Consider the multiple sine-wave function below as an example.
```math
f(x) = \sin(x) + \frac{1}{N}\sin(Nx)
```
Our aim is to demonstrate how to train a model in `RobustNeuralNetworks.jl`, and how to set constraints to ensure the model naturally satisfies some user-defined robustness certificate. We'll follow the steps below to fit an LBDN model to our function ``f(x)``:
1. Generate training data
2. Define a model with a Lipshitz bound (maximum slope) of `1.0`
3. Define a loss function
4. Train the model to minimise the loss function
5. Examine the trained model

## 1. Generate training data

Let's generate training data for ``f(x)`` on the interval ``[0, 2\pi]`` and choose ``N = 5`` as an example. We `zip()` the data up into a sequence of tuples `(x,y)` to make training with `Flux.jl` easier in Step 4.

```@example curve_fit
# Function to estimate
N = 5
f(x) = sin(x)+(1/N)*sin(N*x)

# Training data
dx = 0.1
xs = 0:dx:2π
ys = f.(xs)
data = zip(xs,ys)
```

## 2. Define a model

Since we are only dealing with a simple one-dimensional curve, we can afford to use a small model. Let's choose an LBDN with four hidden layers, each with 15 neurons, and a Lipschitz bound of `γ = 1.0`. This means that the maximum slope the model can achieve between two points will be exactly `1.0` by construction.

```@example curve_fit
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
```

Notice that we first constructed the model parameters `model_ps` defining a (dense) LBDN with [`DenseLBDNParams`](@ref) and then created a callable `model` with the [`DiffLBDN`](@ref) wrapper. In `RobustNeuralNetworks.jl`, model parameterisations are separated from the "explicit" definition of the model used for evaluation on data. The [`DiffLBDN`](@ref) model wrapper combines the two together in a model structure more familiar to [`Flux.jl`](https://fluxml.ai/) users for convenience. See the [Package Overview](@ref) for more information.

## 3. Define a loss function

Let's stick to a simple loss function based on the mean-squared error (MSE) for this example. All [`AbstractLBDN`](@ref) models take an `AbstractArray` as their input, which is why `x` and `y` are wrapped in vectors.
```@example curve_fit
# Loss function
loss(model,x,y) = Flux.mse(model([x]),[y]) 
```

## 4. Train the model

Our objective is to minimise the MSE loss function with a model that has a Lipschitz bound no greater than `1.0`. Let's set up a callback function to check the fit error and slope of our model at each training epoch.

```@example curve_fit
using Flux

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
```

We'll train the model for 200 training epochs a learning rate of `lr = 2e-4`. We'll also use the [`Adam`](https://fluxml.ai/Flux.jl/stable/training/optimisers/#Flux.Optimise.Adam) optimiser from `Flux.jl` and the default [`Flux.train!`](https://fluxml.ai/Flux.jl/stable/training/reference/#Flux.Optimise.train!-NTuple{4,%20Any}) method.

```@example curve_fit
# Define hyperparameters and optimiser
num_epochs = 200
lr = 2e-4
opt_state = Flux.setup(Adam(lr), model)

# Train the model
for i in 1:num_epochs
    Flux.train!(loss, model, data, opt_state)
    (i % 100 == 0) && progress(model, i, xs, ys, dx)
end
```

Note that this training loop is for demonstration only. For a better fit, and indeed on more complex problems, we strongly recommend:
- Increasing the number of training epochs
- Defining your own [training loop](https://fluxml.ai/Flux.jl/stable/training/training/) 
- Using [ParameterSchedulers.jl](https://github.com/FluxML/ParameterSchedulers.jl) to vary the learning rate.

## 5. Examine the trained model

We can now plot the results to see what our model looks like.

```@example curve_fit
using CairoMakie

# Create a figure
f1 = Figure(resolution = (600, 400))
ax = Axis(f1[1,1], xlabel="x", ylabel="y")

ŷ = map(x -> model([x])[1], xs)
lines!(xs, ys, label = "Data")
lines!(xs, ŷ, label = "LBDN")
axislegend(ax)
save("lbdn_curve_fit.svg", f1)
```
![](lbdn_curve_fit.svg)

The model roughly approximates the multiple sine-wave ``f(x)``, but maintains a maximum Lipschitz constant (slope on the graph) below 1. 

```@example curve_fit
# Estimate Lipschitz lower-bound
lip(model, xs, dx) = maximum(abs.(diff(model(xs'), dims=2)))/dx
println("Empirical lower Lipschitz bound: ", round(lip(model, xs, dx); digits=2))
```

The benefit of using an LBDN is that we have full control over the Lipschitz bound, and can still use standard unconstrained gradient descent tools lile `Flux.train!` to train our models. For examples in which setting the Lipschitz bound improves model performance and robustness, see [Image Classification with LBDN](@ref) and [Reinforcement Learning with LBDN](@ref).
