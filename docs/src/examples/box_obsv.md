# Observer Design with REN

*Full example code can be found [here](https://github.com/acfr/RobustNeuralNetworks.jl/blob/main/examples/src/ren_obsv_box.jl).*

In [Reinforcement Learning with LBDN](@ref), we designed a controller for a simple nonlinear system consisting of a box sitting in a tub of fluid, suspended between two springs. We assumed the controller had *full state knowledge*: i.e, it had access to both the position and velocity of the box. In many practical situations, we might only be able to measure some of the system states. For example, our box may have a camera to estimate its position but not its velocity. In these cases, we need a [*state observer*](https://en.wikipedia.org/wiki/State_observer) to estimate the full state of the system for feedback control.

In this example, we will show how a contracting REN can be used to learn stable observers for dynamical systems. A common approach to designing state estimators for nonlinear systems is the *Extended Kalman Filter* ([EKF](https://en.wikipedia.org/wiki/Extended_Kalman_filter)). In our case, we'll consider observer design as a supervised learning problem. For a detailed explanation of the theory behind this example, please refer to Section VIII of [Revay, Wang & Manchester (2021)](https://ieeexplore.ieee.org/document/10179161). 

See [PDE Observer Design with REN](@ref) for explanation of a more complex example from the paper.

## 1. Background theory

Suppose we have a discrete-time, nonlinear dynamical system of the form

```math
\begin{aligned}
x_{t+1} &= f_d(x_t, u_t) \\
y_t &= g_d(x_t, u_t)
\end{aligned}
```
with state vector ``x_t,`` controlled inputs ``u_t,`` and measured outputs ``y_t.`` Our aim is to estimate the sequence ``\{x_0, x_1, \ldots, x_T \}`` over some time period ``[0,T]`` given only the measurements ``y_t`` and inputs ``u_t`` at each time step. We'll use a very general form for an observer
```math
\hat{x}_{t+1} = f_o(\hat{x}_t, u_t, y_t)
```
where ``\hat{x}`` is the state estimate. For those interested, a more common structure is the [Luenberger observer](https://en.wikipedia.org/wiki/State_observer).

We want the observer error to converge to zero as time progresses, or ``\hat{x}_t \rightarrow x_t`` as ``t \rightarrow \infty``. It turns out that our observer only has to satisfy the following two conditions to guarantee this.

1. The observer must be a contracting system (see [Contracting systems](@ref)).
2. The observer must satisfy a "correctness" condition which says that, given perfect knowledge of the state, measurements, and inputs, the observer can exactly predict the next state. Mathematically, we write this as
```math
f_o(x_t,u_t,y_t) = f_d(x_t,u_t).
```
Note the use of ``x_t`` not ``\hat{x}_t`` above. It turns out that if the correctness condition is only approximately satisfied so that ``|f_o(x_t,u_t,y_t) - f_d(x_t,u_t)| < \rho`` for some small number ``\rho``, then the observer error will still be bounded. See Appendix E of the [paper](https://ieeexplore.ieee.org/document/10179161) for details.

Lucky for us, `RobustNeuralNetworks.jl` contains REN models that are guaranteed to be contracting. To learn a stable observer with RENs, all we have to do is minimise the one-step-ahead prediction error. I.e: if we have a batch of data ``z = \{x_i, u_i, y_i, \ i = 1,2,\ldots,N\},`` then we should train our model to minimise the loss function
```math
\mathcal{L}(z, \theta) = \sum_{i=1}^N |f_o(x_i,u_i,y_i) - f_d(x_i,u_i)|^2,
```
where ``\theta`` contains the learnable parameters of the REN.


## 2. Generate training data

Consider the same nonlinear box system we used for [Reinforcement Learning with LBDN](@ref), this time with a measurement function `gd` to give ``y_t = g_d(x_t,u_t)``. We'll assume that only the position of the box is known, so ``y_t = x_t``.

```julia
m = 1                   # Mass (kg)
k = 5                   # Spring constant (N/m)
μ = 0.5                 # Viscous damping coefficient (kg/m)
nx = 2                  # Number of states

# Continuous and discrete dynamics and measurements
_visc(v::Matrix) = μ * v .* abs.(v)
f(x::Matrix,u::Matrix) = [x[2:2,:]; (u[1:1,:] - k*x[1:1,:] - _visc(x[2:2,:]))/m]
fd(x,u) = x + dt*f(x,u)
gd(x::Matrix) = x[1:1,:]
```

We'll assume for this example that the box always starts at rest in a random initial position between ``\pm0.5``m, after which it is released and allowed to oscillate freely with no added forces (so ``u = 0``). Learning an observer typically requires a large amount of training data to capture the behaviour of the system in different scenarios, so we'll consider 200 batches simulating 10s of motion.
```julia
using Random
rng = MersenneTwister(0)

dt = 0.01               # Time-step (s)
Tmax = 10               # Simulation horizon
ts = 1:Int(Tmax/dt)     # Time array indices

batches = 200
u  = fill(zeros(1, batches), length(ts)-1)
X  = fill(zeros(1, batches), length(ts))
X[1] = 0.5*(2*rand(rng, nx, batches) .-1)

for t in ts[1:end-1]
    X[t+1] = fd(X[t],u[t])
end
```
We've stored the states of the system across each batch in `X`. To compute the one-step-ahead loss ``\mathcal{L},`` we'll need to separate this data into the states at the "current" time `Xt` and at the "next" time `Xn,` then compute the measurement outputs.
```julia
Xt = X[1:end-1]
Xn = X[2:end]
y = gd.(Xt)
```
With that done, we store the data for training, shuffling it so there is no bias in the training towards earlier timesteps.
```julia
observer_data = [[ut; yt] for (ut,yt) in zip(u, y)]
indx = shuffle(rng, 1:length(observer_data))
data = zip(Xn[indx], Xt[indx], observer_data[indx])
```

## 3. Define a model

Since we need our model to be a contracting dynamical system, the obvious choice is to use [`ContractingRENParams`](@ref). The inputs to the model are ``[u_t;y_t]``, and its outputs should be the state estimate ``\hat{x}_{t+1}``. The flag `output_map=false` sets the output map of the REN to just return its own internal state. That way, the internal state of the REN is exactly the state estimate ``\hat{x}``.

```julia
using RobustNeuralNetworks

nv = 200
nu = size(observer_data[1], 1)
ny = nx
model_ps = ContractingRENParams{Float64}(nu, nx, nv, ny; output_map=false, rng)
model = DiffREN(model_ps)
```

## 4. Train the model

As mentioned above, our loss function should be the one-step-ahead prediction error of the REN observer. We can write this as follows.
```julia
using Statistics

function loss(model, xn, xt, inputs)
    xpred = model(xt, inputs)[1]
    return mean(sum((xn - xpred).^2, dims=1))
end
```

We've written a function to train the observer that decreases the learning rate by a factor of 10 if the mean gets stuck or starts to increase. The core of this function is just a simple `Flux.jl` training loop. We also report the mean loss at each epoch to inform the user how training is progressing.
```julia
using Flux
using Printf

function train_observer!(model, data; epochs=50, lr=1e-3, min_lr=1e-6)

    opt_state = Flux.setup(Adam(lr), model)
    mean_loss = [1e5]
    for epoch in 1:epochs

        # Gradient descent update
        batch_loss = []
        for (xn, xt, inputs) in data
            train_loss, ∇J = Flux.withgradient(loss, model, xn, xt, inputs)
            Flux.update!(opt_state, model, ∇J[1])
            push!(batch_loss, train_loss)
        end
        @printf "Epoch: %d, Lr: %.1g, Loss: %.4g\n" epoch lr mean(batch_loss)

        # Drop learning rate if mean loss is stuck or growing
        push!(mean_loss, mean(batch_loss))
        if (mean_loss[end] >= mean_loss[end-1]) && !(lr <= min_lr)
            lr = 0.1lr
            Flux.adjust!(opt_state, lr)
        end
    end
    return mean_loss
end
tloss = train_observer!(model, data)
```

## 5. Evaluate the trained model

Now that we've trained the REN observer to minimise the one-step-ahead prediction error, let's see if the observer error actually does converge to zero. First, we'll need some test data. 
```julia
batches   = 50
ts_test   = 1:Int(20/dt)
u_test    = fill(zeros(1, batches), length(ts_test))
x_test    = fill(zeros(nx,batches), length(ts_test))
x_test[1] = 0.2*(2*rand(rng, nx, batches) .-1)

for t in ts_test[1:end-1]
    x_test[t+1] = fd(x_test[t], u_test[t])
end
observer_inputs = [[u;y] for (u,y) in zip(u_test, gd.(x_test))]
```

Next, we'll need a function to simulate the REN observer using its own state ``\hat{x}`` rather than the true system state. We can use the very neat tool [`Flux.Recur`](https://fluxml.ai/Flux.jl/stable/models/layers/#Flux.Recur) for this. We'll assume the observer has no idea what the initial state is, so guess that ``\hat{x}_0 = 0``.
```julia
function simulate(model::AbstractREN, x0, u)
    recurrent = Flux.Recur(model, x0)
    output = recurrent.(u)
    return output
end
x0hat = init_states(model, batches)
xhat = simulate(model, x0hat, observer_inputs)
```

Having simulated the state estimate on the test data, it's time to plot our results. This takes a little bit of setting up to make it look nice, but all the code below is just formatting and plotting.
```julia
using CairoMakie

function plot_results(x, x̂, ts)

    # Observer error
    Δx = x .- x̂

    ts = ts.*dt
    _get_vec(x, i) = reduce(vcat, [xt[i:i,:] for xt in x])
    q   = _get_vec(x,1)
    q̂   = _get_vec(x̂,1)
    qd  = _get_vec(x,2)
    q̂d  = _get_vec(x̂,2)
    Δq  = _get_vec(Δx,1)
    Δqd = _get_vec(Δx,2)

    fig = Figure(resolution = (800, 400))
    ga = fig[1,1] = GridLayout()

    ax1 = Axis(ga[1,1], xlabel="Time (s)", ylabel="Position (m)", title="States")
    ax2 = Axis(ga[1,2], xlabel="Time (s)", ylabel="Position (m)", title="Observer Error")
    ax3 = Axis(ga[2,1], xlabel="Time (s)", ylabel="Velocity (m/s)")
    ax4 = Axis(ga[2,2], xlabel="Time (s)", ylabel="Velocity (m/s)")
    axs = [ax1, ax2, ax3, ax4]

    for k in axes(q,2)
        lines!(ax1, ts,  q[:,k],  linewidth=0.5,  color=:grey)
        lines!(ax1, ts,  q̂[:,k],  linewidth=0.25, color=:red)
        lines!(ax2, ts, Δq[:,k],  linewidth=0.5,  color=:grey)
        lines!(ax3, ts,  qd[:,k], linewidth=0.5,  color=:grey)
        lines!(ax3, ts,  q̂d[:,k], linewidth=0.25, color=:red)
        lines!(ax4, ts, Δqd[:,k], linewidth=0.5,  color=:grey)
    end

    qmin, qmax = minimum(minimum.((q,q̂))), maximum(maximum.((q,q̂)))
    qdmin, qdmax = minimum(minimum.((qd,q̂d))), maximum(maximum.((qd,q̂d)))
    ylims!(ax1, qmin, qmax)
    ylims!(ax2, qmin, qmax)
    ylims!(ax3, qdmin, qdmax)
    ylims!(ax4, qdmin, qdmax)
    xlims!.(axs, ts[1], ts[end])
    display(fig)
    return fig
end
fig = plot_results(x_test, xhat, ts_test)
```
![](../assets/ren-obsv/ren_box_obsv.svg)

In the left-hand panels, grey lines represent the true states of the system, while red lines are for the observer prediction. In the right-hand panels, we see the observer error nicely converging to zero as the observer identifies the correct velocity for all simulation runs. 

It's worth noting that at no point did we directly train the REN to minimise the observer error. This is a natural result of using a model that is guaranteed to be contracting, and training it to minimise the one-step-ahead prediction error. Note that there is still some residual observer error in the velocity, since our observer is only trained to approximately satisfy the correctness condition.