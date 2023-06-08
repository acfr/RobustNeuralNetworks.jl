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


#####################################################################
# Problem setup

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

fd(x,u) = x + dt*f(x + dt*f(x,u)/2,u)
gd(x::Matrix) = x[1:1,:]

# Generate training data
u = zeros(1,  length(ts))
X = zeros(nx, length(ts))
X[:,1] = 0.1*ones(nx)

for t in ts[1:end-1]
    X[:,t+1] = fd(X[:,t], u[:,t])
    u[t+1] = u[t] + 0.01*randn(rng)
end

Xt = X[:, 1:end-1]
Xn = X[:, 2:end]
y = gd(X)

# Store data for training
input_data = [u; y][:,1:end-1]
batchsize = 200
data = Flux.Data.DataLoader((Xn, Xt, input_data); rng, batchsize, shuffle=true)


#####################################################################
# Train a model

# Define a REN model for the observer
nv = 100                     # Works with 100, and works ok with 50 (vel known)
nu = size(input_data, 1)
ny = nx
model_ps = ContractingRENParams{Float64}(nu, nx, nv, ny; is_output=false)
model = DiffREN(model_ps)

# Loss function: one step ahead error (average over time)
function loss(model, xn, xt, inputs)
    xpred = model(xt, inputs)[1]
    return mean(sum((xn - xpred).^2, dims=1))
end

# Train the model
function train_observer!(model, data; epochs=50, lr=1e-3, min_lr=1e-6, verbose=false)

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
        if (mean_loss[end] >= mean_loss[end-1]) && (lr > min_lr)
            lr = 0.1lr
            Flux.adjust!(opt_state, lr)
        end
    end
    return mean_loss
end
tloss = train_observer!(model, data; epochs=50, verbose=true)


#####################################################################
# Generate test data

# Generate test data (a bunch of initial conditions)
batches   = 5
ts_test   = 1:Int(10/dt)
u_test    = fill(zeros(1, batches), length(ts_test))
x_test    = fill(zeros(nx,batches), length(ts_test))
x_test[1] = 0.2*(2*rand(rng, nx, batches) .-1)

for t in ts_test[1:end-1]
    x_test[t+1] = fd(x_test[t], u_test[t])
end
observer_inputs = [[u;y] for (u,y) in zip(u_test, gd.(x_test))]


#####################################################################
# Test one-step-ahead prediction error on test data

xh = fill(zeros(nx,batches), length(ts_test)-1)
for t in ts_test[1:end-1]
    xh[t] = model(x_test[t], observer_inputs[t])[1]
end

a = x_test[2:end]
b = xh
do_diff(at,bt) = mean(sum((at - bt).^2, dims=1))
c = do_diff.(a,b)
@printf "Loss on test data: %.2g\n" mean(c)


#######################################################################
# Simulate observer error

# Simulate the model through time
function simulate(model::AbstractREN, x0, u)
    recurrent = Flux.Recur(model, x0)
    output = recurrent.(u)
    return output
end
x0hat = init_states(model, batches)
xhat = simulate(model, x0hat, observer_inputs)

# Plot results
function plot_results(x, x̂, ts)

    # Observer error
    Δx = x .- x̂

    ts = ts.*dt
    _get_pv(x, i) = reduce(vcat, [xt[i:i,:] for xt in x])
    q   = _get_pv(x,1)
    q̂   = _get_pv(x̂,1)
    qd  = _get_pv(x,2)
    q̂d  = _get_pv(x̂,2)
    Δq  = _get_pv(Δx,1)
    Δqd = _get_pv(Δx,2)

    fig = Figure(resolution = (800, 400))
    ga = fig[1,1] = GridLayout()

    ax1 = Axis(ga[1,1], xlabel="Time (s)", ylabel="Position (m)", title="Actual")
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
    return fig, axs
end
out = plot_results(x_test, xhat, ts_test)
println()