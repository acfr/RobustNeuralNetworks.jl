# Package Overview

The `RobustNeuralNetwork.jl` package is divided into Recurrent Equilibrium Network (REN) and Lipschitz-Bounded Deep Network (LBDN) models.

## REN Overview

The REN models are defined by two fundamental types:

- Any subtype of [`AbstractRENParams`](@ref) holds all the information required to directly parameterise a REN satisfying some user-defined behavioural constraints.

- Any subtype of [`AbstractREN`](@ref) represents the REN in its explicit form so that it can be called and evaluated.

These are described in more detail below.

### (Direct) Parameter Types

Subtypes of [`AbstractRENParams`](@ref) define *direct parameterisations* of a REN. They are not callable models. There are four REN parameter types currently in this package:

- [`ContractingRENParams`](@ref) parameterises a REN with a user-defined upper bound on the contraction rate.

- [`LipschitzRENParams`](@ref) parameterises a REN with a user-defined Lipschitz constant of $\gamma \in (0,\infty)$.

- [`PassiveRENParams`](@ref) parameterises an input/output passive REN with user-tunable passivity parameter $\nu \ge 0$.

- [`GeneralRENParams`](@ref) parameterises a REN satisfying some generalbehavioural constraints defined by an Integral Quadratic Constraint (IQC).

For more information on these four parameterisations, please see [Revay et al. (2021)](https://doi.org/10.48550/arXiv.2104.05942).

Each of these parameter types has the following collection of attributes:

- A static nonlinearity `nl`. Common choices are `Flux.relu` or `Flux.tanh` (see [`Flux.jl`](http://fluxml.ai/Flux.jl/stable/) for more information).

- Model sizes `nu`, `nx`, `nv`, `ny` defining the number of inputs, states, neurons, and outputs (respectively).

- An instance of [`DirectParams`](@ref) containing the direct parameters of the REN, including all **trainable** parameters.

- Other attributes used to define how the direct parameterisation should be converted to the implicit model. These parameters encode the user-tunable behavioural constraints. Eg: $\gamma$ for a Lipschitz-bounded REN.

The typical workflow is to create an instance of a REN parameterisation only once. This defines all dimensions and desired properties of a REN. It is then converted to an explicit model for the REN to be evaluated.



### Explicit REN Models

An *explicit* REN model must be created to call and use the network for computation. The explicit parameterisation contains all information required to evaluate a REN. We encode RENs in explicit form as subtypes of the [`AbstractREN`](@ref) type. Each subtype of `AbstractREN` is callable and includes the following attributes:

- A static nonlinearity `nl` and model sizes `nu`, `nx`, `nv`, `ny` (same as `AbstractRENParams`.

- An instance of `ExplicitParams` containing all REN parameters in explicit form for model evaluation (see the [`ExplicitParams`](@ref) docs for more detail).

Each subtype of `AbstractRENParams` has a method [`direct_to_explicit`](@ref) associated with it that converts the `DirectParams` struct to an instance of `ExplicitParams` satisfying the specified behavioural constraints.

There are three explicit REN wrappers currently implemented in this package. Each of them constructs a REN from a direct parameterisation `ps::AbstractRENParams` and can be used to evaluate REN models.

- [`REN`](@ref) is the basic and most commonly-used wrapper. A new instance of [`REN`](@ref) must be created whenever the parameters `ps` are changed.

- [`WrapREN`](@ref) includes both the `DirectParams` and `ExplicitParams` as part of the REN wrapper. When any of the direct parameters are changed, the explicit model can be updated by calling [`update_explicit!`](@ref).

!!! warning "`WrapREN` incompatible with Flux.jl"
    Since the explicit parameters are stored in an instance of `WrapREN`, changing them with `update_explicit!` directly mutates the model. This will cause errors if the model is to be trained with [`Flux.jl`](http://fluxml.ai/Flux.jl/stable/). Use [`REN`](@ref) or [`DiffREN`](@ref) to avoid this issue.

- [`DiffREN`](@ref) also includes `DirectParams`, but never stores the `ExplicitParams`. Instead, the explicit parameters are computed every time the model is evaluated. This is slow, but does not require creating a new object when the parameters are updated, and is still compatible with `Flux.jl`.

See the docstring of each wrapper and the examples (eg: [PDE Observer Design with REN](@ref)) for more details.


## Walkthrough

Let's step through the example from [Getting Started](@ref), which constructs and evaluates a Lipschitz-bounded REN. Start by importing packages and setting a random seed.

```@example walkthrough
using Random
using RobustNeuralNetworks
```

Let's set a random seed and define our batch size and some hyperparameters. For this example, we'll build a Lipschitz-bounded REN with 4 inputs, 2 outputs, 10 states, 20 neurons, and a Lipschitz bound of `γ = 1`.

```@example walkthrough
rng = MersenneTwister(42)
batches = 10

nu, nx, nv, ny = 4, 10, 20, 2
γ = 1
```

Let's construct the REN parameters. The variable `lipschitz_ren_ps` contains all the parameters required to build a Lipschitz-bounded REN.

```@example walkthrough
lipschitz_ren_ps = LipschitzRENParams{Float64}(nu, nx, nv, ny, γ; rng=rng)
```

Once the parameters are defined, we can create a REN object in its explicit form.

```@example walkthrough
ren = REN(lipschitz_ren_ps)
```

Now we can evaluate the REN. Note that we can use the [`init_states`](@ref) function to create a batch of initial states, all zeros, of the correct dimensions.

```@example walkthrough
# Some random inputs
x0 = init_states(ren, batches; rng=rng)
u0 = randn(rng, ren.nu, batches)

# Evaluate the REN over one timestep
x1, y1 = ren(x0, u0)
```

Having evaluated the REN, we can check that the outputs are the same as in the original example.

```@example walkthrough
# Print results for testing
yout = round.(y1; digits=2)
println(yout[1,:])
println(yout[2,:])
```


## LBDN Overview

*[To be written once LBDN has been properly added to the package.]*
