# Package Overview

The `RobustNeuralNetwork.jl` package is divided into Recurrent Equilibrium Network (REN) and Lipschitz-Bounded Deep Network (LBDN) models.

## REN Overview

The REN models are composed of two fundamental types:

- Any subtype of [`AbstractRENParams`](@ref) holds all the information required to directly parameterise a REN satisfying some user-defined behavioural constraints.

- Any subtype of [`AbstractREN`](@ref) represents the REN in its explicit form so that it can be called and used like a typical neural network model.

These are described in more detail below.

### (Direct) Parameter Types

Subtypes of [`AbstractRENParams`](@ref) define a *direct parameterisation* of a REN. They are not callable models. There are four REN parameter types currently in this package:

- [`ContractingRENParams`](@ref) encodes the information required to build a contracting REN with a user-defined upper bound on the contraction rate.

- [`LipschitzRENParams`](@ref) encodes the information required to build a REN with a user-defined Lipschitz constant of $\gamma \in (0,\infty)$.

- [`PassiveRENParams`](@ref) encodes the information required to build an input/output passive REN with user-tunable passivity parameter $\nu \ge 0$.

- [`GeneralRENParams`](@ref) encodes the information required to build a REN satisfying some generalbehavioural constraints defined by an Integral Quadratic Constraint (IQC).

For more information on these four parameterisations, please see [this paper](https://doi.org/10.48550/arXiv.2104.05942).

The attributes of a subtype of [`AbstractRENParams`](@ref) are divided as follows:

- A static nonlinearity `nl`. Common choices are `Flux.relu` or `Flux.tanh` (see [`Flux.jl`](http://fluxml.ai/Flux.jl/stable/) for more information).

- Model sizes `nu`, `nx`, `nv`, `ny` defining the number of inputs, states, neurons, and outputs (respectively).

- An instance of [`DirectParams`](@ref) containing the direct (implicit) parameterisation of the REN.

- Other attributes used to define how the direct parameterisation should be converted to the implicit model. These parameters encode the user-tunable behavioural constraints.

The typical workflow is to create an instance of a REN parameterisation once which defines all dimensions and desired properties of a REN. This is then converted to an explicit model when the REN is called.



### Explicit REN Models

An *explicit* REN model must be created to call and use the network for computation. The explicit parameterisation contains all information required to evaluate a REN. We encode RENs in explicit form as subtypes of [`AbstractREN`](@ref). Each subtype of [`AbstractREN`](@ref) is a callable type with attributes:

- A static nonlinearity `nl`.

- Model sizes `nu`, `nx`, `nv`, `ny`.

- An instance of [`ExplicitParams`](@ref) containing all REN parameters in explicit form for implementation (see the docs for [`ExplicitParams`](@ref) for more detail).

Each subtype of [`AbstractRENParams`](@ref) has a method [`direct_to_explicit`](@ref) associated with it that converts the [`DirectParams`](@ref) struct to an instance of [`ExplicitParams`](@ref) satisfying the user-defined behavioural constraints.

There are three explicit REN wrappers currently in this package:

- [`REN`](@ref): ...

- [`WrapREN`](@ref): ...

- [`DiffREN`](@ref): ...

See the docstrings for each wrapper for more details.

[TODO:] Edit this and make it sound better

### Walkthrough

Go through the WrapREN walkthrough.



## LBDN Overview

*[To be written once LBDN has been properly added to the package.]*


<!-- 
### Non-differentiable REN Wrapper

... Will want to explain about REN wrappers properly. This needs to be clear...

There are many ways to train a REN, some of which do not involve differentiating the model. In these cases, it is convenient to have a wrapper `WrapREN <: AbstractREN` for the `REN` type that does not need to be destroyed an recreated whenever the direct parameters change. `WrapREN` is structured exactly the same as `REN`, but also holds the `AbstractRENParams` used to construct its explicit model. The explicit model can be updated in-place following any changes to the direct parameters. See below for an example.

[NOTE] This operation is not compatible with Flux differentiation because the explicit parameters are mutated during the update. -->