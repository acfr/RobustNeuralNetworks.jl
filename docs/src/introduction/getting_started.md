# Getting Started

## Installation

`RobustNeuralNetworks.jl` is written in Julia and can be installed with the package manager. To add the package, type the following into the REPL.

```
] add RobustNeuralNetworks
```

## Basic Usage

You should now be able to construct robust neural network models. The following example constructs a Lipschitz-bounded REN and evalutates it given a batch of random initial states and inputs.

```jldoctest
using Random
using RobustNeuralNetworks

# Setup
rng = MersenneTwister(42)
batches = 10
nu, nx, nv, ny = 4, 10, 20, 1
γ = 1

# Construct a REN
lipschitz_ren_ps = LipschitzRENParams{Float64}(nu, nx, nv, ny, γ; rng=rng)
ren = REN(lipschitz_ren_ps)

# Some random inputs
x0 = init_states(ren, batches; rng=rng)
u0 = randn(rng, ren.nu, batches)

# Evaluate the REN over one timestep
x1, y1 = ren(x0, u0)

# Print results for testing
println(round.(y1; digits=2))

# output

[0.23 -0.01 -0.06 0.15 -0.03 -0.11 0.0 0.42 0.24 0.22]
```

For detailed examples of training models from `RobustNeuralNetworks.jl`, we recommend starting with [Fitting a Curve with LBDN](@ref) and working through the subsequent examples.


## Walkthrough

Let's step through the example above. It constructs and evaluates a Lipschitz-bounded REN. We start by importing packages and setting a random seed.

```@example walkthrough
using Random
using RobustNeuralNetworks
```

Let's set a random seed and define our batch size and some hyperparameters. For this example, we'll build a Lipschitz-bounded REN with 4 inputs, 1 output, 10 states, 20 neurons, and a Lipschitz bound of `γ = 1`.

```@example walkthrough
rng = MersenneTwister(42)
batches = 10

γ = 1
nu, nx, nv, ny = 4, 10, 20, 1
```

Now we can construct the REN parameters. The variable `lipschitz_ren_ps` contains all the parameters required to build a Lipschitz-bounded REN. Note that we separate the model parameterisation and its "explicit" (callable) form in `RobustNeuralNetworks.jl`. See the [Package Overview](@ref) for more details.

```@example walkthrough
lipschitz_ren_ps = LipschitzRENParams{Float64}(nu, nx, nv, ny, γ; rng=rng)
```

Once the parameters are defined, we can create a REN object in its explicit form.

```@example walkthrough
ren = REN(lipschitz_ren_ps)
```

Now we can evaluate the REN. We can use the [`init_states`](@ref) function to create a batch of initial states, all zeros, of the correct dimensions.

```@example walkthrough
# Some random inputs
x0 = init_states(ren, batches)
u0 = randn(rng, ren.nu, batches)

# Evaluate the REN over one timestep
x1, y1 = ren(x0, u0)
```

Having evaluated the REN, we can check that the outputs are the same as in the original example.

```@example walkthrough
# Print results for testing
println(round.(y1; digits=2))
```