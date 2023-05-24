# Package Overview OLD

The `RobustNeuralNetwork.jl` package is divided into Recurrent Equilibrium Network (REN) and Lipschitz-Bounded Deep Network (LBDN) models.


## REN Overview



!!! info "Separate Objects for Parameters and Model"
    When working with most models (eg: [RNN](https://fluxml.ai/Flux.jl/stable/models/layers/#Flux.RNN) and [LSTM](https://fluxml.ai/Flux.jl/stable/models/layers/#Flux.LSTM)) the typical workflow is to create a single instance of a model. Its parameters are updated during training, but the model object is only created once. For example:

    ```julia
    using Flux

    # Define a model
    model = Flux.RNNCell(2,5)

    # Train the model
    for k in 1:num_training_epochs
        ...                     # Run some code and compute gradients
        Flux.update!(...)       # Update model parameters
    ```

    When working with RENs, it is much more efficient to split up the model parameterisation and the model implementation into subtypes of [`AbstractRENParams`](@ref) and [`AbstractREN`](@ref). Converting our direct parameterisation to an explicit model for evaluation can be slow, so we only do it when the model parameters are updated:

    ```julia
    using Flux
    using RobustNeuralNetworks

    # Define a model parameterisation
    params = ContractingRENParams{Float64}(2, 5, 10, 1)

    # Train the model
    for k in 1:num_training_epochs
        model = REN(params)     # Create explicit model for evaluation
        ...                     # Run some code and compute gradients
        Flux.update!(...)       # Update model parameters
    ```
    See the section on [REN Wrappers](@ref) for more details.

### (Direct) Parameter Types

The typical workflow is to create an instance of a REN parameterisation only once. This defines all dimensions and desired properties of a REN. It is then converted to an explicit model for the REN to be evaluated.


#### REN Wrappers

There are three explicit REN wrappers currently implemented in this package. Each of them constructs a REN from a direct parameterisation `params::AbstractRENParams` and can be used to evaluate REN models.

- [`REN`](@ref) is the basic and most commonly-used wrapper. A new instance of [`REN`](@ref) must be created whenever the parameters `params` are changed.

!!! tip "REN is recommended"
    We strongly recommend using `REN` to train your models with `Flux.jl`. It is the most efficient subtype of `AbstractREN` that is compatible with automatic differentiation.

- [`WrapREN`](@ref) includes both the `DirectRENParams` and `ExplicitRENParams` as part of the REN wrapper. When any of the direct parameters are changed, the explicit model can be updated by calling [`update_explicit!`](@ref). This can be useful when not using automatic differentiation to train the model. For example:

```julia
using RobustNeuralNetworks

# Define a model parameterisation AND a model
params = ContractingRENParams{Float64}(2, 5, 10, 1)
model  = WrapREN(params)

# Train the model
for k in 1:num_training_epochs
    ...                     # Run some code and compute gradients
    ...                     # Update model parameters
    update_explicit!(model) # Update explicit model parameters
```

!!! warning "WrapREN incompatible with Flux.jl"
    Since the explicit parameters are stored in an instance of `WrapREN`, changing them with `update_explicit!` directly mutates the model. This will cause errors if the model is to be trained with [`Flux.jl`](http://fluxml.ai/Flux.jl/stable/). Use [`REN`](@ref) or [`DiffREN`](@ref) to avoid this issue.

- [`DiffREN`](@ref) also includes `DirectRENParams`, but never stores the `ExplicitRENParams`. Instead, the explicit parameters are computed every time the model is evaluated. This is slow, but does not require creating a new object when the parameters are updated, and is still compatible with `Flux.jl`. For example:

```julia
using Flux

# Define a model parameterisation AND a model
params = ContractingRENParams{Float64}(2, 5, 10, 1)
model  = DiffREN(params)

# Train the model
for k in 1:num_training_epochs
    ...                     # Run some code and compute gradients
    Flux.update!(...)       # Update model parameters
```

See the docstring of each wrapper and the examples (eg: [PDE Observer Design with REN](@ref)) for more details.
