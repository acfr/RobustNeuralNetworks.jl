cd(@__DIR__)
using Pkg
Pkg.activate("..")

# TODO: Tidy up the unnecessary packages
using Distributions
using Flux
using Flux.Optimise:update!
using Flux.Optimise
using Formatting
using LinearAlgebra
using Plots
using Random
using RobustNeuralNetworks
using Zygote


# TODO: Get this working with Float32, will be faster
# TODO: Would be even better to get it working on the GPU

# Problem setup
nx = 51             # Number of states
n_in = 1            # Number of inputs
L = 10.0            # Size of spatial domain


# Generate data from finite difference approximation of heat equation
function reaction_diffusion_equation(;L=10.0, steps=5, nx=51, sigma=0.1)

    # Discretise space/time
    dx = L / (nx - 1)
    dt = sigma * dx^2

    # State dynamics
    function f(u0, d)
        u, un = copy(u0), copy(u0)
        for _ in 1:steps
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

    # Output function
    g(u, d) = [d; u[end ÷ 2:end ÷ 2, :]]
    return f, g
end
f, g = reaction_diffusion_equation()

nPoints = Int(1e5)
X = zeros(nx, nPoints)
U = zeros(n_in, nPoints)
for t in 1:nPoints - 1
    X[:, t + 1:t + 1] = f(X[:, t:t], U[:, t:t])
    
    # Calculate next u
    u_next = U[1,t] .+ 0.05f0 * randn(Float64)
    if u_next > 1
        u_next = 1
    elseif u_next < 0
        u_next = 0
    end
    U[:,t + 1] .= u_next
end
xt = X[:, 1:end - 1]
xn = X[:, 2:end]
y = g(X, U)

input_data = [U; y][:, 1:end - 1]  # inputs to observer
batchsize = 20

data = Flux.Data.DataLoader((xn, xt, input_data), batchsize=batchsize, shuffle=true)

# Model parameters
nv = 500
nu = size(input_data, 1)
ny = nx

# Constuction REN
model_params = ContractingRENParams{Float64}(
    nu, nx, nv, ny; 
    nl = tanh,
    ϵ=0.01,
    polar_param = false, 
    is_output = false
)
model = DiffREN(model_params)

function train_observer!(model, data, opt; Epochs=200, regularizer=nothing, solve_tol=1E-5, min_lr=1E-7)
    ps = Flux.params(model)
    mean_loss = [1E5]
    loss_std = []
    for epoch in 1:Epochs
        batch_loss = []
        for (xni, xi, ui) in data
            function calc_loss()
                xpred = model(xi, ui)[1]
                return mean(norm(xpred[:, i] - xni[:, i]).^2 for i in 1:size(xi, 2))
            end

            train_loss, back = Zygote.pullback(calc_loss, ps)

            # Calculate gradients and update loss
            ∇J = back(one(train_loss))
            update!(opt, ps, ∇J)
        
            push!(batch_loss, train_loss)
            printfmt("Epoch: {1:2d}\tTraining loss: {2:1.4E} \t lr={3:1.1E}\n", epoch, train_loss, opt.eta)
        end

        # Print stats through epoch
        println("------------------------------------------------------------------------")
        printfmt("Epoch: {1:2d} \t mean loss: {2:1.4E}\t std: {3:1.4E}\n", epoch, mean(batch_loss), std(batch_loss))
        println("------------------------------------------------------------------------")
        push!(mean_loss, mean(batch_loss))
        push!(loss_std, std(batch_loss))

        # Check for decrease in loss.
        if mean_loss[end] >= mean_loss[end - 1]
            println("Reducing Learning rate")
            opt.eta *= 0.1
            if opt.eta <= min_lr  # terminate optim.
                return mean_loss, loss_std
            end
        end
    end
    return mean_loss, loss_std
end

opt = Flux.Optimise.ADAM(1E-3)
tloss, loss_std = train_observer!(model, data, opt; Epochs=200, min_lr=1E-7)

# Test observer
T = 1000
time = 1:T

u = ones(Float64, n_in, length(time)) / 2
x = ones(Float64, nx, length(time))

for t in 1:T - 1
    x[:, t + 1] = f(x[:, t:t], u[t:t])
    
    # Calculate next u
    u_next = u[t] + 0.05f0 * (randn(Float64))
    if u_next > 1
        u_next = 1
    elseif u_next < 0
        u_next = 0
    end
    u[t + 1] = u_next
end
y = [g(x[:, t:t], u[t]) for t in time]

batches = 1
observer_inputs = [repeat([ui; yi], outer=(1, batches)) for (ui, yi) in zip(u, y)]
# println(typeof(observer_inputs),size(observer_inputs))

# Foward simulation

unzip(a) = (getfield.(a, x) for x in fieldnames(eltype(a)))
function simulate(model::AbstractREN, x0, u)
    eval_cell = (x, u) -> model(x, u)
    recurrent = Flux.Recur(eval_cell, x0)
    output = [recurrent(input) for input in u]
    return output
end

x0 = zeros(nx, batches)
xhat = simulate(model, x0, observer_inputs)
# xhat = collect(simulate(testdata["model"], cpu(x0), cpu(observer_inputs)))[1]

p1 = heatmap(x, color=:cividis, aspect_ratio=1);

Xhat = reduce(hcat, xhat)
p2 = heatmap(Xhat[:, 1:batches:end], color=:cividis, aspect_ratio=1);
p3 = heatmap(abs.(x - Xhat[:, 1:batches:end]), color=:cividis, aspect_ratio=1);

p = plot(p1, p2, p3; layout=(3, 1))
# savefig(p,"pde_observer.png")