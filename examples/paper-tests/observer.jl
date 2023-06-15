cd(@__DIR__)
using Pkg
Pkg.activate("../")

m = 1                   # Mass (kg)
k = 5                   # Spring constant (N/m)
μ = 0.5                 # Viscous damping (kg/m)
nx = 2                  # Number of states

# Continuous and discrete dynamics and measurements
f(x::Matrix,u::Matrix) = [x[2:2,:]; (u[1:1,:] - 
    k*x[1:1,:] - μ*x[2:2,:].^2)/m]
fd(x,u) = x + dt*f(x,u)
gd(x::Matrix) = x[1:1,:]

Tmax = 10               # Simulation horizon
dt = 0.01               # Time-step (s)
ts = 1:Int(Tmax/dt)     # Time array indices

nbatch = 200
u = fill(zeros(1, nbatch), length(ts)-1)
X = fill(zeros(1, nbatch), length(ts))
X[1] = 0.5*(2*rand(nx, nbatch) .- 1)

for t in ts[1:end-1]
    X[t+1] = fd(X[t],u[t])
end

using Random

# Current/next state, measurements
Xt = X[1:end-1]
Xn = X[2:end]
y  = gd.(Xt)

# Store training data
obsv_data = [[ut; yt] for (ut,yt) in zip(u, y)]
indx = shuffle(1:length(obsv_data))
data = zip(Xn[indx], Xt[indx], obsv_data[indx])

using RobustNeuralNetworks

T  = Float64
nv = 100
nu = size(obsv_data[1], 1)
ny = nx
model_ps = ContractingRENParams{T}(
    nu, nx, nv, ny; is_output=false)
model = DiffREN(model_ps)

using Statistics

function loss(model, xn, xt, inputs)
    xpred = model(xt, inputs)[1]
    return mean(sum((xn - xpred).^2, dims=1))
end

using Flux

function train_observer!(
    model, data; 
    epochs=2, lr=1e-3, min_lr=1e-4
)
    opt_state = Flux.setup(Adam(lr), model)
    mean_loss = [1e5]
    for epoch in 1:epochs

        # Gradient descent update
        batch_loss = []
        for (xn, xt, inputs) in data
            tloss, dJ = Flux.withgradient(
                loss, model, xn, xt, inputs)
            Flux.update!(opt_state, model, dJ[1])
            push!(batch_loss, tloss)
        end

        # Reduce lr if loss is stuck or growing
        push!(mean_loss, mean(batch_loss))
        if (mean_loss[end] >= mean_loss[end-1]) && 
           (lr > min_lr)
            lr *= 0.1
            Flux.adjust!(opt_state, lr)
        end
    end
    return mean_loss
end
tloss = train_observer!(model, data)

nbatch    = 50
ts_test   = 1:Int(10/dt)
u_test    = fill(zeros(1, nbatch), length(ts_test))
x_test    = fill(zeros(nx,nbatch), length(ts_test))
x_test[1] = 0.2*(2*rand(nx, nbatch) .-1)

for t in ts_test[1:end-1]
    x_test[t+1] = fd(x_test[t], u_test[t])
end
y_test = gd.(x_test)
obsv_in = [[u;y] for (u,y) in zip(u_test, y_test)]

function simulate(model::AbstractREN, x0, u)
    recurrent = Flux.Recur(model, x0)
    output = recurrent.(u)
    return output
end
x0hat = zeros(model.nx, nbatch)
xhat = simulate(model, x0hat, obsv_in)