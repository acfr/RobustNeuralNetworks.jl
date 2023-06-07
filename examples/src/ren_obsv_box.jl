cd(@__DIR__)
using Pkg
Pkg.activate("..")

using CairoMakie
using Flux
using Printf
using Random
using RobustNeuralNetworks
using Statistics

rng = MersenneTwister(42)


# -------------------------
# Problem setup
# -------------------------

# System parameters
m = 1                   # Mass (kg)
k = 5                   # Spring constant (N/m)
μ = 0.5                 # Viscous damping coefficient (kg/m)
nx = 2                  # Number of states
dt = 0.001              # Time-step (s)
Tmax = 100              # Simulation horizon
ts = 1:Int(Tmax/dt)     # Time array indices

# Continuous and discrete dynamics and measurements
f(x::Vector,u::Vector) = [x[2]; (u[1] - k*x[1] - μ*x[2].^2)/m]
f(x::Matrix,u::Matrix) = [x[2:2,:]; (u[1:1,:] - k*x[1:1,:] - μ*x[2:2,:].^2)/m]
fd(x,u) = x + dt*f(x,u)
# fd(x,u) = x + dt*f(x + dt*f(x,u)/2,u)

gd(x::Vector) = x[1]
gd(x::Matrix) = x[1:1,:]

# Generate test data
function get_data(ts; σu=5, init=zeros)

    npoints = length(ts)
    X = init(nx, npoints)
    U = σu*randn(rng, 1, npoints)

    for t in 1:npoints-1
        X[:,t+1] = fd(X[:,t], U[:,t])
    end
    return X, U
end

X, u = get_data(ts)
Xt = X[:, 1:end-1]
Xn = X[:, 2:end]
y = gd(X)

# Store data for training
input_data = [u; y][:,1:end-1]
batchsize = 200
data = Flux.Data.DataLoader((Xn, Xt, input_data); rng, batchsize, shuffle=true)

# Define a REN model for the observer
nv = 10
nu = size(input_data, 1)
ny = nx
model_ps = ContractingRENParams{Float64}(nu, nx, nv, ny; nl=tanh, is_output=false)
model = DiffREN(model_ps)

# Loss function: one step ahead error (average over time)
function loss(model, xn, xt, inputs)
    xpred = model(xt, inputs)[1]
    return mean(sum((xn - xpred).^2, dims=1))
end

# Train the model
function train_observer!(model, data; epochs=50, lr=1e-3, min_lr=1e-7, verbose=false)

    opt_state = Flux.setup(Adam(lr), model)
    mean_loss = [1e5]
    for epoch in 1:epochs

        batch_loss = []
        for (xn, xt, inputs) in data
            train_loss, ∇J = Flux.withgradient(loss, model, xn, xt, inputs)
            Flux.update!(opt_state, model, ∇J[1])
            push!(batch_loss, train_loss)
        end
        verbose && @printf "Epoch: %d, Lr: %.1g, Loss: %.4g\n" epoch lr mean(batch_loss)

        # Drop learning rate if mean loss is stuck or growing
        push!(mean_loss, mean(batch_loss))
        if mean_loss[end] >= mean_loss[end-1]
            lr *= 0.1
            Flux.adjust!(opt_state, lr)
            (lr <= min_lr) && (return mean_loss)
        end
    end
    return mean_loss
end
tloss = train_observer!(model, data; epochs=100, verbose=true)

# Generate test data: a bunch of initial conditions
function test_data(ts, batches=20)

    x_test = fill(zeros(nx,batches), length(ts))
    x_test[1] = randn(rng, nx, batches)
    u_test = fill(zeros(1, batches), length(ts))

    for t in 1:length(ts)-1
        x_test[t+1] = fd(x_test[t], u_test[t])
    end
    input_test = [[u;y] for (u,y) in zip(u_test,gd.(x_test))]
    return x_test, input_test
end

t_test = 1:Int(10/dt)
x_test, input_test = test_data(t_test, 50)
x_pred = [model(x, u)[1] for (x,u) in zip(x_test, input_test)]
Δx_test = x_test[2:end] .- x_pred[1:end-1]


# TODO: This is not quite right yet. I need to plot on same axes scales, and probably roll out the learned REN itself instead of one-step-ahead prediction. TEST THIS AND CHECK.

# Plot the states for now
function plot_results(x, Δx, ts)

    ts1 = ts[1:end-1]
    _get_pv(x, i) = reduce(vcat, [xt[i:i,:] for xt in x])
    pos = _get_pv(x,1)
    vel = _get_pv(x,2)

    Δpos = _get_pv(Δx,1)
    Δvel = _get_pv(Δx,2)

    fig = Figure(resolution = (800, 400))
    ga = fig[1,1] = GridLayout()

    ax1 = Axis(ga[1,1], xlabel="Time (s)", ylabel="Position (m)", title="Actual")
    ax2 = Axis(ga[1,2], xlabel="Time (s)", ylabel="Position (m)", title="Observer Error")
    ax3 = Axis(ga[2,1], xlabel="Time (s)", ylabel="Velocity (m/s)")
    ax4 = Axis(ga[2,2], xlabel="Time (s)", ylabel="Velocity (m/s)")

    for k in axes(pos,2)
        lines!(ax1, ts.*dt,   pos[:,k], linewidth=0.5, color=:grey)
        lines!(ax2, ts1.*dt, Δpos[:,k], linewidth=0.5, color=:grey)
        lines!(ax3, ts.*dt,   vel[:,k], linewidth=0.5, color=:grey)
        lines!(ax4, ts1.*dt, Δvel[:,k], linewidth=0.5, color=:grey)
    end

    display(fig)
    return fig
end
plot_results(x_test, Δx_test, t_test)
println()