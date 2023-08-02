# Fitting a Curve with LBDN

*Full example code can be found [here](https://github.com/acfr/RobustNeuralNetworks.jl/blob/main/examples/src/lbdn_curvefit.jl).*

For our first example, let's fit a Lipschitz-bounded Deep Network (LBDN) to a curve in one dimension. Consider the step function function below.
```math
f(x) = 
\begin{cases}
1 \ \text{if} \ x > 0 \\ 0  \ \text{if} \ x < 0
\end{cases}
```
Our aim is to demonstrate how to train a model in `RobustNeuralNetworks.jl`, and how to ensure the model naturally satisfies some user-defined robustness certificate (the Lipschitz bound). We'll follow the steps below to fit an LBDN model to our function ``f(x)``:
1. Generate training data
2. Define a model with a Lipshitz bound (maximum slope) of `10.0`
3. Define a loss function
4. Train the model to minimise the loss function
5. Examine the trained model

## 1. Generate training data

Let's generate training data for ``f(x)`` on the interval ``[-0.3, 0.3]`` as an example. We `zip()` the data up into a sequence of tuples `(x,y)` to make training with `Flux.jl` easier in Step 4.

```@example curve_fit
# Function to estimate
f(x) = x < 0 ? 0 : 1

# Training data
dx = 0.01
xs = -0.3:dx:0.3
ys = f.(xs)
data = zip(xs,ys)
```

## 2. Define a model

Since we are only dealing with a simple one-dimensional curve, we can afford to use a small model. Let's choose an LBDN with four hidden layers, each with 16 neurons, and a Lipschitz bound of `γ = 10.0`. This means that the maximum slope the model can achieve between two points should be exactly `10.0` by construction.

```@example curve_fit
using Random
using RobustNeuralNetworks

# Random seed for consistency
rng = Xoshiro(0)

# Model specification
nu = 1                  # Number of inputs
ny = 1                  # Number of outputs
nh = fill(16,4)         # 4 hidden layers, each with 16 neurons
γ = 10                  # Lipschitz bound of 10

# Set up model: define parameters, then create model
model_ps = DenseLBDNParams{Float64}(nu, nh, ny, γ; rng)
model = DiffLBDN(model_ps)
```

Note that we first constructed the model parameters `model_ps`, and *then* created a callable `model`. In `RobustNeuralNetworks.jl`, model parameterisations are separated from "explicit" definitions of a model used for evaluation on data. See the [Direct & explicit parameterisations](@ref) for more information.

!!! info "A layer-wise approach"
    We have also provided single LBDN layers with [`SandwichFC`](@ref). Introduced in [Wang & Manchester (2023)](https://proceedings.mlr.press/v202/wang23v.html), the [`SandwichFC`](@ref) layer is a fully-connected or dense layer with a guaranteed Lipschitz bound of 1.0. We have designed the user interface for [`SandwichFC`](@ref) to be as similar to that of [`Flux.Dense`](https://fluxml.ai/Flux.jl/stable/models/layers/#Flux.Dense) as possible. This may be more convenient for users used to working with `Flux.jl`.

    For example, we can construct an identical model to the LBDN `model` above with the following.
    ```julia
    using Flux

    chain_model = Flux.Chain(
        (x) -> (√γ * x),
        SandwichFC(nu => nh[1], Flux.relu; T=Float64, rng),
        SandwichFC(nh[1] => nh[2], Flux.relu; T=Float64, rng),
        SandwichFC(nh[2] => nh[3], Flux.relu; T=Float64, rng),
        SandwichFC(nh[3] => nh[4], Flux.relu; T=Float64, rng),
        (x) -> (√γ * x),
        SandwichFC(nh[4] => ny; output_layer=true, T=Float64, rng),
    )
    ```

    See Section 3.1 of [Wang & Manchester (2023)](https://proceedings.mlr.press/v202/wang23v.html) for further details.

## 3. Define a loss function

Let's stick to a simple loss function based on the mean-squared error (MSE) for this example. All [`AbstractLBDN`](@ref) models take an `AbstractArray` as their input, which is why `x` and `y` are wrapped in vectors.
```@example curve_fit
# Loss function
loss(model,x,y) = Flux.mse(model([x]),[y]) 
```

## 4. Train the model

Our objective is to minimise the loss function with a model that has a Lipschitz bound no greater than `10.0`. Let's set up a callback function to check the fit error and slope of our model at each training epoch.

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

We'll train the model for 300 training epochs a learning rate of `lr = 2e-4`. We'll also use the [`Adam`](https://fluxml.ai/Flux.jl/stable/training/optimisers/#Flux.Optimise.Adam) optimiser from `Flux.jl` and the default [`Flux.train!`](https://fluxml.ai/Flux.jl/stable/training/reference/#Flux.Optimise.train!-NTuple{4,%20Any}) method.

```@example curve_fit
# Define hyperparameters and optimiser
num_epochs = 300
lr = 2e-4
opt_state = Flux.setup(Adam(lr), model)

# Train the model
for i in 1:num_epochs
    Flux.train!(loss, model, data, opt_state)
    (i % 100 == 0) && progress(model, i, xs, ys, dx)
end
```

Note that this training loop is for demonstration only. For a better fit, or on more complex problems, we strongly recommend:
- Increasing the number of training epochs
- Defining your own [training loop](https://fluxml.ai/Flux.jl/stable/training/training/) 
- Using [ParameterSchedulers.jl](https://github.com/FluxML/ParameterSchedulers.jl) to vary the learning rate.

## 5. Examine the trained model

The final estimated lower bound of our Lipschitz constantt is very close to the maximum allowable value of 10.0.
```@example curve_fit
using Printf

# Estimate Lipschitz lower-bound
Empirical_Lipschitz = lip(model, xs, dx)
@printf "Imposed Lipschitz upper bound:   %.2f\n" get_lipschitz(model)
@printf "Empirical Lipschitz lower bound: %.2f\n" Empirical_Lipschitz
```

We can now plot the results to see what our model looks like.

```@example curve_fit
using CairoMakie

# Create a figure
f1 = Figure(resolution = (600, 400))
ax = Axis(f1[1,1], xlabel="x", ylabel="y")

# Compute the best-possible fit with Lipschitz bound 10.0
get_best(x) = x<-0.05 ? 0 : (x<0.05 ? 10x + 0.5 : 1)
ybest = get_best.(xs)
ŷ = map(x -> model([x])[1], xs)

# Plot
lines!(xs, ys, label = "Data")
lines!(xs, ybest, label = "Max. slope = 10.0")
lines!(xs, ŷ, label = "LBDN slope = $(round(Empirical_Lipschitz; digits=2))")
axislegend(ax, position=:lt)
save("lbdn_curve_fit.svg", f1)
```
![](lbdn_curve_fit.svg)

The model roughly approximates the step function ``f(x)``, but maintains a maximum Lipschitz constant (slope on the graph) below 10.0. It is reasonably close to the best-possible value, and can easily be improved with a slightly larger model and more training time.

The benefit of using an LBDN is that we have full control over the Lipschitz bound, and can still use standard unconstrained gradient descent tools lile `Flux.train!` to train our models. For examples in which setting the Lipschitz bound improves model performance and robustness, see [Image Classification with LBDN](@ref) and [Reinforcement Learning with LBDN](@ref).
