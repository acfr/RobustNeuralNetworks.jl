# This file is a part of RobustNeuralNetworks.jl. License is MIT: https://github.com/acfr/RobustNeuralNetworks.jl/blob/main/LICENSE 

cd(@__DIR__)
using Pkg
Pkg.activate("..")

using CairoMakie
using CUDA
using Flux
using Printf
using Random
using RobustNeuralNetworks
using Statistics

"""
A note for the interested reader:

- Change `dev = gpu` and `T = Float32` to train the REN observer on an Nvidia GPU with CUDA
- This example is currently not optimised for the GPU, and runs faster on CPU
- It would be easy to re-write it to be much faster on the GPU
- If you feel like doing this, please go ahead and submit a pull request :)

"""

rng = MersenneTwister(0)
dev = cpu
T = Float64


#####################################################################
# Problem setup

# System parameters
m = 1                   # Mass (kg)
k = 5                   # Spring constant (N/m)
μ = 0.5                 # Viscous damping coefficient (kg/m)
nx = 2                  # Number of states

# Continuous and discrete dynamics and measurements
_visc(v) = μ * v .* abs.(v)
f(x,u) = [x[2:2,:]; (u[1:1,:] - k*x[1:1,:] - _visc(x[2:2,:]))/m]
fd(x,u) = x + dt*f(x,u)
gd(x) = x[1:1,:]

# Generate training data
dt = T(0.01)            # Time-step (s)
Tmax = 10               # Simulation horizon
ts = 1:Int(Tmax/dt)     # Time array indices

batches = 200
u  = fill(zeros(T, 1, batches), length(ts)-1)
X  = fill(zeros(T, 1, batches), length(ts))
X[1] = (2*rand(rng, T, nx, batches) .- 1) / 2

for t in ts[1:end-1]
    X[t+1] = fd(X[t],u[t])
end

Xt = X[1:end-1]
Xn = X[2:end]
y = gd.(Xt)

# Store data for training
observer_data = [[ut; yt] for (ut,yt) in zip(u, y)]
indx = shuffle(rng, 1:length(observer_data))
data = zip(Xn[indx] |> dev, Xt[indx] |> dev, observer_data[indx]|> dev)


#####################################################################
# Train a model

# Define a REN model for the observer
nv = 200
nu = size(observer_data[1], 1)
ny = nx
model_ps = ContractingRENParams{Float32}(nu, nx, nv, ny; output_map=false, rng)
model = DiffREN(model_ps) |> dev

# Loss function: one step ahead error (average over time)
function loss(model, xn, xt, inputs)
    xpred = model(xt, inputs)[1]
    return mean(sum((xn - xpred).^2, dims=1))
end

# Train the model
function train_observer!(model, data; epochs=50, lr=1e-3, min_lr=1e-6)

    opt_state = Flux.setup(Adam(lr), model)
    mean_loss = [T(1e5)]
    for epoch in 1:epochs

        batch_loss = []
        for (xn, xt, inputs) in data
            train_loss, ∇J = Flux.withgradient(loss, model, xn, xt, inputs)
            Flux.update!(opt_state, model, ∇J[1])
            push!(batch_loss, train_loss)
        end
        @printf "Epoch: %d, Lr: %.1g, Loss: %.4g\n" epoch lr mean(batch_loss)

        # Drop learning rate if mean loss is stuck or growing
        push!(mean_loss, mean(batch_loss))
        if (mean_loss[end] >= mean_loss[end-1]) && !(lr < min_lr || lr ≈ min_lr)
            lr = 0.1lr
            Flux.adjust!(opt_state, lr)
        end
    end
    return mean_loss
end
tloss = train_observer!(model, data)


#####################################################################
# Generate test data

# Generate test data (a bunch of initial conditions)
batches   = 50
ts_test   = 1:Int(20/dt)
u_test    = fill(zeros(1, batches), length(ts_test))
x_test    = fill(zeros(nx,batches), length(ts_test))
x_test[1] = 0.2*(2*rand(rng, nx, batches) .-1)

for t in ts_test[1:end-1]
    x_test[t+1] = fd(x_test[t], u_test[t])
end
observer_inputs = [[u;y] for (u,y) in zip(u_test, gd.(x_test))]


#######################################################################
# Simulate observer error

# Simulate the model through time
function simulate(model::AbstractREN, x0, u)
    recurrent = Flux.Recur(model, x0)
    output = recurrent.(u)
    return output
end
x0hat = init_states(model, batches)
xhat = simulate(model, x0hat |> dev, observer_inputs |> dev)

# Plot results
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

    fig = Figure(resolution = (600, 400))
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
fig = plot_results(x_test, xhat |> cpu, ts_test)
save("../results/ren-obsv/ren_box_obsv.svg", fig)
