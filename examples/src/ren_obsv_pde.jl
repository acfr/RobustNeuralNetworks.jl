# This file is a part of RobustNeuralNetworks.jl. License is MIT: https://github.com/acfr/RobustNeuralNetworks.jl/blob/main/LICENSE 

cd(@__DIR__)
using Pkg
Pkg.activate("..")

using BSON
using CairoMakie
using Flux
using Formatting
using LinearAlgebra
using Random
using RobustNeuralNetworks
using Statistics


# TODO: Do this with Float32, will be faster
# TODO: Would be even better to get it working on the GPU. For later...
dtype = Float64

# Problem setup
nx = 51             # Number of states
n_in = 1            # Number of inputs
L = 10.0            # Size of spatial domain
sigma = 0.1         # Used to construct time step

# Discretise space and time
dx = L / (nx - 1)
dt = sigma * dx^2

# State dynamics and output functions f, g
function f(u0, d)
    u, un = copy(u0), copy(u0)
    for _ in 1:5
        u = copy(un) 

        # FD approximation of heat equation
        f_local(v) = v[2:end - 1, :] .* (1 .- v[2:end - 1, :]) .* ( v[2:end - 1, :] .- 0.5)
        laplacian(v) = (v[1:end - 2, :] + v[3:end, :] - 2v[2:end - 1, :]) / dx^2
        
        # Euler step for time
        un[2:end - 1, :] = u[2:end - 1, :] + dt * (laplacian(u) + f_local(u) / 2 )

        # Boundary condition
        un[1:1, :] = d;
        un[end:end, :] = d;
    end
    return u
end

g(u, d) = [d; u[end ÷ 2:end ÷ 2, :]]

# Generate simulated data
function get_data(npoints=1000; init=zeros)

    X = init(dtype, nx, npoints)
    U = init(dtype, n_in, npoints)

    for t in 1:npoints-1

        # Next state
        X[:, t+1] = f(X[:, t], U[:, t])
        
        # Next input bₜ
        u_next = U[t] + 0.05f0*randn(dtype)
        (u_next > 1) && (u_next = 1)
        (u_next < 0) && (u_next = 0)
        U[t + 1] = u_next
    end
    return X, U
end

X, U = get_data(100000; init=zeros)
xt = X[:, 1:end - 1]
xn = X[:, 2:end]
y = g(X, U)

# Store for the observer (inputs are inputs to observer)
input_data = [U; y][:, 1:end - 1]
batches = 200
data = Flux.Data.DataLoader((xn, xt, input_data), batchsize=batches, shuffle=true)

# Constuct a REN
# TODO: Test if we actually need all of this
# TODO: Does it matter what ϵ, polar_param, or nl are?
nv = 500
nu = size(input_data, 1)
ny = nx
model_params = ContractingRENParams{dtype}(
    nu, nx, nv, ny; 
    nl = tanh, ϵ=0.01,
    polar_param = false, 
    output_map = false
)
model = DiffREN(model_params) # (see the documentation)

# Define a loss function
function loss(model, xn, x, u)
    xpred = model(x, u)[1]
    return mean(norm(xpred[:, i] - xn[:, i]).^2 for i in 1:size(x, 2))
end

# Train the model
function train_observer!(model, data; Epochs=50, lr=1e-3, min_lr=1e-7)

    # Set up the optimiser
    opt_state = Flux.setup(Adam(lr), model)

    mean_loss, loss_std = [1e5], []
    for epoch in 1:Epochs
        batch_loss = []
        for (xni, xi, ui) in data

            # Get gradient and store loss
            train_loss, ∇J = Flux.withgradient(loss, model, xni, xi, ui)
            Flux.update!(opt_state, model, ∇J[1])
        
            # Store losses for later
            push!(batch_loss, train_loss)
            printfmt("Epoch: {1:2d}\tTraining loss: {2:1.4E} \t lr={3:1.1E}\n", epoch, train_loss, lr)
        end

        # Print stats through epoch
        println("------------------------------------------------------------------------")
        printfmt("Epoch: {1:2d} \t mean loss: {2:1.4E}\t std: {3:1.4E}\n", epoch, mean(batch_loss), std(batch_loss))
        println("------------------------------------------------------------------------")
        push!(mean_loss, mean(batch_loss))
        push!(loss_std, std(batch_loss))

        # Check for decrease in loss
        if mean_loss[end] >= mean_loss[end - 1]
            println("Reducing Learning rate")
            lr *= 0.1
            Flux.adjust!(opt_state, lr)
            (lr <= min_lr) && (return mean_loss, loss_std)
        end
    end
    return mean_loss, loss_std
end

# Train and save the model
tloss, loss_std = train_observer!(model, data; Epochs=50, lr=1e-3, min_lr=1e-7)
bson("../results/ren-obsv/pde_obsv.bson", 
    Dict(
        "model" => model, 
        "training_loss" => tloss, 
        "loss_std" => loss_std
    )
)

# Test observer
T = 2000
init = (args...) -> 0.5*ones(args...)
x, u = get_data(T, init=init)
y = [g(x[:, t:t], u[t]) for t in 1:T]

batches = 1
observer_inputs = [repeat([ui; yi], outer=(1, batches)) for (ui, yi) in zip(u, y)]

# Simulate the model through time
function simulate(model::AbstractREN, x0, u)
    recurrent = Flux.Recur(model, x0)
    output = recurrent.(u)
    return output
end
x0 = init_states(model, batches)
xhat = simulate(model, x0, observer_inputs)
Xhat = reduce(hcat, xhat)

# Make a plot to show PDE and errors
function plot_heatmap(f1, xdata, i)

    # Make and label the plot
    xlabel = i < 3 ? "" : "Time steps"
    ylabel = i == 1 ? "True" : (i == 2 ? "Observer" : "Error")
    ax, _ = heatmap(f1[i,1], xdata', colormap=:thermal, axis=(xlabel=xlabel, ylabel=ylabel))

    # Format the axes
    ax.yticksvisible = false
    ax.yticklabelsvisible = false
    if i < 3
        ax.xticksvisible = false
        ax.xticklabelsvisible = false
    end
    xlims!(ax, 0, T)
end

f1 = Figure(resolution=(500,400))
plot_heatmap(f1, x, 1)
plot_heatmap(f1, Xhat[:, 1:batches:end], 2)
plot_heatmap(f1, abs.(x - Xhat[:, 1:batches:end]), 3)
Colorbar(f1[:,2], colorrange=(0,1),colormap=:thermal)

display(f1)
save("../results/ren-obsv/ren_pde.png", f1) # Note: this takes a long time...
