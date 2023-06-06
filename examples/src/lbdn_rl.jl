cd(@__DIR__)
using Pkg
Pkg.activate("..")

using CairoMakie
using Flux
using Flux: glorot_normal
using Printf
using Random
using RobustNeuralNetworks
using Statistics
using Zygote: Buffer

rng = MersenneTwister(42)

# -------------------------
# Problem setup
# -------------------------

# System parameters
m = 1                   # Mass (kg)
k = 5                   # Spring constant (N/m)
μ = 0.5                 # Friction damping coefficient (kg/m)

# Simulation horizon and timestep (s)
Tmax = 4
dt = 0.02
ts = 1:Int(Tmax/dt)

# Start at zero, random goal states
nx, nref, batches = 2, 1, 80
x0 = zeros(nx, batches)
xref = 2*rand(rng, nref, batches) .- 1
uref = k*xref

# Continuous  and discrete dynamics
f(x::Matrix,u::Matrix) = [x[2:2,:]; (u[1:1,:] - k*x[1:1,:] - μ*x[2:2,:].^2)/m]
fd(x::Matrix,u::Matrix) = x + dt*f(x,u)

# Simulate the system given initial condition and a controller
# Controller of the form u = k([x; xref])
function rollout(model, x0, xref; ϵ=0.0)
    z = Buffer([zero([x0;xref])], length(ts))
    x = x0
    for t in ts

        u = model([x;xref])
        z[t] = vcat(x,u)

        u = u + ϵ*(2*rand(rng, size(u)...) .- 1)
        x = fd(x,u)
    end
    return copy(z)
end

# Cost function for z = [x;u] at each time/over all times
weights = [10,1,0.1]
function _cost(z, xref, uref)
    Δz = z .- [xref; zero(xref); uref]
    return mean(sum(weights .* Δz.^2; dims=1))
end
cost(z::AbstractVector) = mean(_cost.(z, (xref,), (uref,)))


# -------------------------
# Train LBDN
# -------------------------

# Define an LBDN model 
nu = nx + nref          # Inputs (states and reference)
ny = 1                  # Outputs (control action u)
nh = fill(32, 2)        # Hidden layers
γ = 20                  # Lipschitz bound
model_ps = DenseLBDNParams{Float64}(nu, nh, ny, γ; nl=relu, rng)

# Choose a loss function
function loss(model_ps, x0, xref)
    model = LBDN(model_ps)
    z = rollout(model, x0, xref)
    return cost(z)
end

# Train the model
function train_box_ctrl!(model, loss_func; lr=1e-3, epochs=250, verbose=false)

    costs = Vector{Float64}()
    opt_state = Flux.setup(Adam(lr), model)

    for k in 1:epochs

        train_loss, ∇J = Flux.withgradient(loss_func, model, x0, xref)
        Flux.update!(opt_state, model, ∇J[1])

        push!(costs, train_loss)
        verbose && @printf "Iter %d loss: %.2f\n" k train_loss
    end

    return costs
end

train_box_ctrl!(model_ps, loss; epochs=1); # Dummy run for just-in-time compiler
t_lbdn = @elapsed (costs = train_box_ctrl!(model_ps, loss))

# -------------------------
# Test LBDN
# -------------------------

# Evaluate final model on an example
model = LBDN(model_ps)
xtest = reshape([1.0],1,1)
ztest = rollout(model, zeros(nx,1), xtest)

# Plot position, velocity, and control input over time
function plot_box_learning(costs, z, xref, indx=1)

    x = [z[t][1,indx] for t in ts]
    v = [z[t][2,indx] for t in ts]
    u = [z[t][3,indx] for t in ts]

    xr = xref[indx]
    ur = k*xr

    f1 = Figure(resolution = (500, 400))
    ga = f1[1,1] = GridLayout()

    ax0 = Axis(ga[1,1], xlabel="Training epochs", ylabel="Cost")
    ax1 = Axis(ga[1,2], xlabel="Time steps", ylabel="Position (m)", )
    ax2 = Axis(ga[2,1], xlabel="Time steps", ylabel="Velocity (m/s)")
    ax3 = Axis(ga[2,2], xlabel="Time steps", ylabel="Control (N)")

    lines!(ax0, costs, color=:black)
    lines!(ax1, ts, x, color=:black)
    lines!(ax2, ts, v, color=:black)
    lines!(ax3, ts, u, color=:black)

    lines!(ax1, ts, xr*ones(size(ts)), color=:red, linestyle=:dash)
    lines!(ax2, ts, zeros(size(ts)), color=:red, linestyle=:dash)
    lines!(ax3, ts, ur*ones(size(ts)), color=:red, linestyle=:dash)

    display(f1)
    return f1
end

fig = plot_box_learning(costs, ztest, xtest)
save("../results/lbdn_rl.svg", fig) 


# -------------------------
# Compare to DiffLBDN
# -------------------------
model_ps2 = DenseLBDNParams{Float64}(nu, nh, ny, γ; nl=relu, rng)
model2 = DiffLBDN(model_ps2)

loss2(model, x0, xref) = cost(rollout(model, x0, xref))

train_box_ctrl!(model2, loss2; epochs=1); # Dummy run for just-in-time compiler
t_diff_lbdn = @elapsed (costs2 = train_box_ctrl!(model2, loss2))


# -------------------------
# Compare to Dense
# -------------------------

# Train new model with smaller γ
initb(n) = glorot_normal(rng,n)
model3 = Chain(
    Dense(nu => nh[1], relu; init=glorot_normal, bias=initb(nh[1])),
    Dense(nh[1] => nh[2], relu; init=glorot_normal, bias=initb(nh[2])),
    Dense(nh[2] => ny; init=glorot_normal, bias=initb(ny)),
)
train_box_ctrl!(model3, loss2; lr=5e-3) # Dummy run for just-in-time compiler
t_dense = @elapsed (costs3 = train_box_ctrl!(model3, loss2; lr=5e-3))

# Test robustness
ϵ = 10.0
ztest  = rollout(model, zeros(nx,1), xtest; ϵ)
ztest3 = rollout(model3, zeros(nx,1), xtest; ϵ)
plot_box_learning(costs, ztest, xtest)
plot_box_learning(costs3, ztest3, xtest)

@printf "Training time for LBDN:     %.2fs\n" t_lbdn
@printf "Training time for DiffLBDN: %.2fs\n" t_diff_lbdn
@printf "Training time for Dense:    %.2fs\n" t_dense
