# Package Overview

The `RobustNeuralNetwork.jl` package contains two classes of neural network models: Recurrent Equilibrium Networks (RENs) and Lipschitz-Bounded Deep Networks (LBDNs). This page will give a brief overview of the model architectures and their parameterisations. 

For more detail on RENs and LBDNs, please see [Revay, Wang & Manchester (2021)](https://doi.org/10.48550/arXiv.2104.05942) and [Wang & Manchester (2023)](https://doi.org/10.48550/arXiv.2301.11526), respectively.


## What are RENs and LBDNs?

A *Recurrent Equilibrium Network* (REN) is a linear system in a feedback interconnection with a nonlinear activation function. Denote ``x_t \in \mathbb{R}^n`` as the internal states of the system, ``u_t \in\mathbb{R}^m`` as its inputs, and ``y_t \in \mathbb{R}^p`` as its outputs. Mathematically, a REN can be represented as

```math
\begin{aligned}
\begin{bmatrix}
x_{t+1} \\ v_t \\ y_t
\end{bmatrix}&=
\overset{W}{\overbrace{
		\left[
		\begin{array}{c|cc}
		A & B_1 & B_2 \\ \hline 
		C_{1} & D_{11} & D_{12} \\
		C_{2} & D_{21} & D_{22}
		\end{array} 
		\right]
}}
\begin{bmatrix}
x_t \\ w_t \\ u_t
\end{bmatrix}+
\overset{b}{\overbrace{
		\begin{bmatrix}
		b_x \\ b_v \\ b_y
		\end{bmatrix}
}}, \\
w_t=\sigma(&v_t):=\begin{bmatrix}
\sigma(v_{t}^1) & \sigma(v_{t}^2) & \cdots & \sigma(v_{t}^q)
\end{bmatrix}^\top, 
\end{aligned}
```

where ``v_t, w_t \in \mathbb{R}^q`` are the inputs and outputs of neurons passing through the activation function ``\sigma``. Graphically, this is equivalent to the following, where the linear (actually affine) system ``G`` represents the first of the two equations above.

```@raw html
<p align="center">
  <img src="../assets/ren.svg" width="35%"/>
</p>
```

A *Lipschitz-Bounded Deep Network* (LBDN) can be thought of as a specialisation of a REN with a state dimension of ``n = 0``. That is, LBDN models have no dynamics or memory associated with them. In reality, we use this simplification to construct LBDN models completely differently to RENs. We construct LBDNs as ``L``-layer feed-forward networks, much like [MLPs](https://en.wikipedia.org/wiki/Multilayer_perceptron) or [CNNs](https://en.wikipedia.org/wiki/Convolutional_neural_network), described by the following recursive equations.

```math
\begin{aligned}
z_0 &= x \\
z_{k+1} &= \sigma(W_k z_k + b_k), \quad k = 0, \ldots, L-1 \\
y &= W_L z_L + b_L
\end{aligned}
```

See [Wang & Manchester (2023)](https://doi.org/10.48550/arXiv.2301.11526) for more details on LBDNs.


## Direct & explicit parameterisations

The key advantage of the models in `RobustNeuralNetworks.jl` is that they naturally satisfy a set of user-defined robustness constraints (see [Robustness metrics and IQCs](@ref)). This means that we can guarantee the robustness of our neural networks *by construction*. There is no need to impose additional (possibly computationally-expensive) constraints while training a REN or LBDN. One can simply use unconstrained optimisation methods like gradient descent and be sure that the final model will satisfy the robustness requirements.

We achieve this by constructing the weight matrices and bias vectors in the models to automatically satisfy some specific linear matrix inequalities (see [Revay, Wang & Manchester (2021)](https://doi.org/10.48550/arXiv.2104.05942) for details). The *learnable parameters* of a model are a set of free variables ``\theta \in \mathbb{R}^N`` which are completely unconstrained. When the set of learnable parameters for a model is exactly ``\mathbb{R}^N`` like this, we call it a **direct parameterisation**. The equations above describe the **explicit parameterisation** of RENs and LBDNs. I.e: a callable model that we can evaluate on data. For a REN, the *explicit parameters* are ``\bar{\theta} = [W, b]``, and for an LBDN they are ``\bar{\theta} = [W_0, b_0, \ldots, W_L, b_L]``.

In `RobustNeuralNetworks.jl`, RENs are defined by two fundamental types. Any subtype of [`AbstractRENParams`](@ref) holds all the information required to directly parameterise a REN satisfying some robustness properties. For example, to initialise the direct parameters of a contracting REN with 1 input, 10 states, 20 neurons, and 1 output, we would use the following.

```@example build_ren
using RobustNeuralNetworks

nu, nx, nv, ny = 1, 10, 20, 1
model_params = ContractingRENParams{Float64}(nu, nx, nv, ny)

typeof(model_params) <: AbstractRENParams
```

Subtypes of [`AbstractREN`](@ref) represent RENs in their explicit form so that they can be called and evaluated. The conversion from the direct to explicit parameters ``\theta \mapsto \bar{\theta}`` is performed when the REN is constructed.

```@example build_ren
model = REN(model_params)

typeof(model) <: AbstractREN
```

The same is true for [`AbstractLBDNParams`](@ref) and [`AbstractLBDN`](@ref) regarding LBDN models.


### Types of direct parameterisations

There are currently four REN parameterisations implemented in this package:

- [`ContractingRENParams`](@ref) parameterises a REN with a user-defined upper bound on the contraction rate.

- [`LipschitzRENParams`](@ref) parameterises a REN with a user-defined Lipschitz constant of $\gamma \in (0,\infty)$.

- [`PassiveRENParams`](@ref) parameterises an input/output passive REN with user-tunable passivity parameter $\nu \ge 0$.

- [`GeneralRENParams`](@ref) parameterises a REN satisfying some general behavioural constraints defined by an Integral Quadratic Constraint (IQC) with parameters (Q,S,R).

Similarly, subtypes of [`AbstractLBDNParams`](@ref) define the direct parameterisation of an LBDN. There is currently only one version implemented in `RobustNeuralNetworks.jl`:

- [`DenseLBDNParams`](@ref) parameterises an dense (fully-connected) LBDN. A dense LBDN is effectively a Lipschitz-bounded [`Flux.Dense`](https://fluxml.ai/Flux.jl/stable/models/layers/#Flux.Dense) network.

See [Robustness metrics and IQCs](@ref) for an explanation of these types.


### Explicit model wrappers

When training a REN or LBDN, we learn and update the direct parameters ``\theta`` and convert them to the explicit parameters ``\bar{\theta}`` only for model evaluation. The main constructors for explicit models are [`REN`](@ref) and [`LBDN`](@ref).

Users familiar with [`Flux.jl`](https://fluxml.ai/) will be used to creating a model just once and then training/updating it on their data. The typical workflow is something like this.

```julia
using Flux
using Random

# Define a model and a loss function
model = Chain(Flux.Dense(1 => 10, Flux.relu), Flux.Dense(10 => 1), Flux.relu)
loss(model, x, y) = Flux.mse(model(x), y)

# Set up some dummy training data
batches = 20
xs, ys = rand(1,batches), rand(1,batches)
data = [(xs, ys)]

# Train the model for 50 epochs
opt_state = Flux.setup(Adam(0.01), model)
for _ in 1:50
	Flux.train!(loss, model, data, opt_state)
end
```

When using a model constructed from [`REN`](@ref) or [`LBDN`](@ref), we need to differentiate through the mapping from direct (learnable) parameters to the explicit model. We therefore need a setup where the model construction is actually part of the loss function. Here's an example with an [`LBDN`](@ref).

```julia
using Flux
using Random

# Define a model parameterisation and a loss function
model_params = DenseLBDNParams{Float64}(1, [10], 1)
function loss(model_params, x, y) 
	model = LBDN(model_params)
	Flux.mse(model(x), y)
end

# Set up some dummy training data
batches = 20
xs, ys = rand(1,batches), rand(1,batches)
data = [(xs, ys)]

# Train the model for 50 epochs
opt_state = Flux.setup(Adam(0.01), model_params)
for _ in 1:50
	Flux.train!(loss, model_params, data, opt_state)
end
```


### Separating parameters and models is efficient

You might ask: why not write a wrapper which just computes the explicit parameters each time the model is called? That would save the user the worry of keeping track of the `model_params` and `model` separately. 

In fact, we have. See [`DiffREN`](@ref), [`DiffLBDN`](@ref), and [`SandwichFC`](@ref). Any model created with these interfaces can be used exactly the same way as a regular [`Flux.jl`](https://fluxml.ai/) model (no need for model construction in the loss function). This is illustrated in examples like [Fitting a Curve with LBDN](@ref) and [Image Classification with LBDN](@ref).

The reason we nominally keep the `model_params` and `model` separate with [`REN`](@ref) and [`LBDN`](@ref) is because the computational bottleneck is converting from the direct to explicit parameters (mapping ``\theta \mapsto \bar{\theta}``). Direct parameters are stored in `model_params`, while explicit parameters are computed when the `model` is created and are stored in it. Using our earlier example of a contracting REN:

```@example build_ren
println(typeof(model_params.direct))
println(typeof(model.explicit))
```

In some applications (eg: reinforcement learning or system identification), a model is called many times with the same explicit parameters ``\bar{\theta}`` before its learnable parameters ``\theta`` are updated. It's therefore efficient to store the explicit parameters, use them many times, and then update them only when the learnable parameters change. We can't store the direct and explicit parameters in the same `model` object because [`Flux.jl`] does not permit mutation of non-learnable parameters. Instead, we separate the two.

!!! info "Which wrapper should I use?"
	Model wrappers like [`DiffREN`](@ref), [`DiffLBDN`](@ref), and [`SandwichFC`](@ref) re-compute the explicit parameters every time the model is called. They are the most convenient choice for applications where the learnable parameters are updated each time the model is called (eg: image classification, curve fitting, etc.). 
	
	For applications where the model is called many times (eg: in a feedback loop) before updating it, use [`REN`](@ref) and [`LBDN`](@ref). They compute the explicit model when constructed and store it for later use, making them more efficient.


## Robustness metrics and IQCs

There are a number of different robustness criteria which our REN models can satisfy. Some relate to the internal dynamics of the model, others relate to the input/output map. LBDNs are less general. They are specifically constructed to satisfy Lipschitz bounds. See the section on [Lipschitz bounds (smoothness)](@ref) below.

### Contracting systems

First and foremost, **all** of our REN models are **contracting systems**. This means that they exponentially "forget" initial conditions. That is, if two copies of the system start at two different initial conditions but are given the same inputs, the internal states will converge over time. See below for an example of a contracting REN with a single internal state.

```@raw html
<p align="center">
  <img src="../assets/contracting_ren.svg" width="50%"/>
</p>
```

### Integral quadratic constraints

We define additional robustness criteria on the input/output map of our RENs with *incremental integral quadratic constraints* (IQCs). Suppose we have a model ``\mathcal{M}`` starting at two different initial conditions ``a,b`` with two different input signals ``u, v``, and consider their corresponding output trajectories ``y^a = \mathcal{M}_a(u)`` and ``y^b = \mathcal{M}_b(v).`` The model ``\mathcal{M}`` satisfies the IQC defined by matrices ``(Q, S, R)`` if

```math
\sum_{t=0}^T
\begin{bmatrix}
y^a_t - y^b_t \\ u_t - v_t
\end{bmatrix}^\top
\begin{bmatrix}
Q & S^\top \\ S & R
\end{bmatrix}
\begin{bmatrix}
y^a_t - y^b_t \\ u_t - v_t
\end{bmatrix} 
\ge -d(a,b)
```

for some function ``d(a,b) \ge 0`` with ``d(a,a) = 0``, where ``0 \preceq Q \in \mathbb{R}^{p\times p}``, ``S\in\mathbb{R}^{m\times p},`` ``R=R^\top \in \mathbb{R}^{m\times m}.`` 

In general, the IQC matrices could be chosen (or optimised) to meet a range of performance criteria. There are a few special cases that are worth noting.

#### Lipschitz bounds (smoothness)

If ``Q = -\frac{1}{\gamma}I``, ``R = \gamma I``, ``S = 0``, the model ``\mathcal{M}`` satisfies an an ``\ell^2`` Lipschitz bound (incremental ``\ell^2``-gain bound) of ``\gamma``.

```math
\|\mathcal{M}_a(u) - \mathcal{M}_b(v)\|_T \le \gamma \|u - v\|_T
```

Qualitatively, the Lipschitz bound is a measure of how smooth the network is. If the Lipschitz bound ``\gamma`` is small, then small changes in the inputs ``u,v`` will lead to small changes in the model output. If ``\gamma`` is large, then the model output might change significantly for even small changes to the inputs. This makes the model more sensitive to noise, adversarial attacks, and other input disturbances.

As the name suggests, the LBDN models are all constructed to have a user-tunable Lipschitz bound.

#### Incremental passivity

There are two cases to consider here. In both cases, the network must have the same number of inputs and outs.

- If ``Q = 0, R = -2\nu I, S = I`` where ``\nu \ge 0``, the model is incrementally passive (incrementally strictly input passive if ``\nu > 0``). Mathematically, the following inequality holds.
```math
\langle \mathcal{M}_a(u) - \mathcal{M}_b(v), u-v \rangle_T \ge \nu \| u-v\|^2_T
```

- If ``Q = -2\rho I, R = 0, S = I`` where ``\rho > 0``, the model is incrementally strictly output passive. Mathematically, the following inequality holds.
```math
\langle \mathcal{M}_a(u) - \mathcal{M}_b(v), u-v \rangle_T \ge \rho \| \mathcal{M}_a(u) - \mathcal{M}_b(v)\|^2_T
```

For more details on IQCs and their use in RENs, please see [Revay, Wang & Manchester (2021)](https://doi.org/10.48550/arXiv.2104.05942).
