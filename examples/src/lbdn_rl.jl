# This file is a part of RobustNeuralNetworks.jl. License is MIT: https://github.com/acfr/RobustNeuralNetworks.jl/blob/main/LICENSE 

cd(@__DIR__)
using Pkg
Pkg.activate("..")

using CairoMakie
using Flux
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
μ = 0.5                 # Viscous damping coefficient (kg/m)

# Simulation horizon and timestep (s)
Tmax = 4
dt = 0.02
ts = 1:Int(Tmax/dt)

# Start at zero, random goal states
nx, nref, batches = 2, 1, 80
x0 = zeros(nx, batches)
qref = 2*rand(rng, nref, batches) .- 1
uref = k*qref

# Continuous and discrete dynamics
_visc(v::Matrix) = μ * v .* abs.(v)
f(x::Matrix,u::Matrix) = [x[2:2,:]; (u[1:1,:] - k*x[1:1,:] - _visc(x[2:2,:]))/m]
fd(x::Matrix,u::Matrix) = x + dt*f(x,u)

# Simulate the system given initial condition and a controller
# Controller of the form u = k([x; qref])
function rollout(model, x0, qref)
    z = Buffer([zero([x0;qref])], length(ts))
    x = x0
    for t in ts
        u = model([x;qref])
        z[t] = vcat(x,u)
        x = fd(x,u)
    end
    return copy(z)
end

# Cost function for z = [x;u] at each time/over all times
weights = [10,1,0.1]
function _cost(z, qref, uref)
    Δz = z .- [qref; zero(qref); uref]
    return mean(sum(weights .* Δz.^2; dims=1))
end
cost(z::AbstractVector, qref, uref) = mean(_cost.(z, (qref,), (uref,)))


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
function loss(model_ps, x0, qref, uref)
    model = LBDN(model_ps)
    z = rollout(model, x0, qref)
    return cost(z, qref, uref)
end

# Train the model
function train_box_ctrl!(model_ps, loss_func; lr=1e-3, epochs=250, verbose=false)

    costs = Vector{Float64}()
    opt_state = Flux.setup(Adam(lr), model_ps)

    for k in 1:epochs

        train_loss, ∇J = Flux.withgradient(loss_func, model_ps, x0, qref, uref)
        Flux.update!(opt_state, model_ps, ∇J[1])

        push!(costs, train_loss)
        verbose && @printf "Iter %d loss: %.2f\n" k train_loss
    end

    return costs
end

costs = train_box_ctrl!(model_ps, loss; verbose=true)


# -------------------------
# Test LBDN
# -------------------------

# Evaluate final model on an example
lbdn = LBDN(model_ps)
x0_test = zeros(2,100)
qr_test = 2*rand(rng, 1, 100) .- 1
z_lbdn = rollout(lbdn, x0_test, qr_test)

# Plot position, velocity, and control input over time
function plot_box_learning(costs, z, qr)

    _get_vec(x, i) = reduce(vcat, [xt[i:i,:] for xt in x])
    q = _get_vec(z, 1)
    v = _get_vec(z, 2)
    u = _get_vec(z, 3)
    t = dt*ts
    
    Δq = q .- qr .* ones(length(z), length(qr_test))
    Δu = u .- k*qr .* ones(length(z), length(qr_test))

    fig = Figure(resolution = (600, 400))
    ga = fig[1,1] = GridLayout()

    ax0 = Axis(ga[1,1], xlabel="Training epochs", ylabel="Cost")
    ax1 = Axis(ga[1,2], xlabel="Time (s)", ylabel="Position error (m)", )
    ax2 = Axis(ga[2,1], xlabel="Time (s)", ylabel="Velocity (m/s)")
    ax3 = Axis(ga[2,2], xlabel="Time (s)", ylabel="Control error (N)")

    lines!(ax0, costs, color=:black)
    for k in axes(q,2)
        lines!(ax1, t, Δq[:,k], linewidth=0.5,  color=:grey)
        lines!(ax2, t,  v[:,k], linewidth=0.5,  color=:grey)
        lines!(ax3, t, Δu[:,k], linewidth=0.5,  color=:grey)
    end

    lines!(ax1, t, zeros(size(t)), color=:red, linestyle=:dash)
    lines!(ax2, t, zeros(size(t)), color=:red, linestyle=:dash)
    lines!(ax3, t, zeros(size(t)), color=:red, linestyle=:dash)
    
    xlims!.((ax1,ax2,ax3), (t[1],), (t[end],))
    display(fig)
    return fig
end

fig = plot_box_learning(costs, z_lbdn, qr_test)
save("../results/lbdn-rl/lbdn_rl.svg", fig)


# ---------------------------------
# Compare to DiffLBDN
# ---------------------------------

# Loss function for differentiable model
loss2(model, x0, qref, uref) = cost(rollout(model, x0, qref), qref, uref)

function lbdn_compute_times(n; epochs=100)

    print("Training models with nh = $n... ")
    lbdn_ps = DenseLBDNParams{Float64}(nu, [n], ny, γ; nl=relu, rng)
    diff_lbdn = DiffLBDN(deepcopy(lbdn_ps))

    t_lbdn = @elapsed train_box_ctrl!(lbdn_ps, loss; epochs)
    t_diff_lbdn = @elapsed train_box_ctrl!(diff_lbdn, loss2; epochs)

    println("Done!")
    return [t_lbdn, t_diff_lbdn]

end

# Evaluate computation time with different hidden-layer sizes
# Run it once first for just-in-time compiler
sizes = 2 .^ (1:9)
lbdn_compute_times(2; epochs=1)
comp_times = reduce(hcat, lbdn_compute_times.(sizes))

# Plot the results
fig = Figure(resolution = (500, 300))
ax = Axis(
    fig[1,1], 
    xlabel="Hidden layer size", 
    ylabel="Training time (s) (100 epochs)", 
    xscale=Makie.log2, yscale=Makie.log10
)
lines!(ax, sizes, comp_times[1,:], label="LBDN")
lines!(ax, sizes, comp_times[2,:], label="DiffLBDN")

xlims!(ax, [sizes[1], sizes[end]])
axislegend(ax, position=:lt)
display(fig)
save("../results/lbdn-rl/lbdn_rl_comptime.svg", fig)
