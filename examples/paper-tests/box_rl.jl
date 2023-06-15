cd(@__DIR__)
using Pkg
Pkg.activate("../")

m = 1                   # Mass (kg)
k = 5                   # Spring constant (N/m)
μ = 0.5                 # Viscous damping (kg/m)

Tmax = 4                # Simulation horizon (s)
dt = 0.02               # Time step (s)
ts = 1:Int(Tmax/dt)     # Array of time indices

nx, nref, batches = 2, 1, 80
x0 = zeros(nx, batches)
qref = 2*rand(nref, batches) .- 1
uref = k*qref

f(x::Matrix,u::Matrix) = [x[2:2,:]; (u[1:1,:] - 
    k*x[1:1,:] - μ*x[2:2,:].^2)/m]
fd(x::Matrix,u::Matrix) = x + dt*f(x,u)

using Zygote: Buffer

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

using Statistics

weights = [10,1,0.1]
function _cost(z, qref, uref)
    Δz = z .- [qref; zero(qref); uref]
    return mean(sum(weights .* Δz.^2; dims=1))
end
cost(z::AbstractVector, qref, uref) = 
    mean(_cost.(z, (qref,), (uref,)))

using RobustNeuralNetworks

T  = Float64
γ  = 20                 # Lipschitz bound
nu = nx + nref          # Inputs (x and reference)
ny = 1                  # Outputs (control action)
nh = fill(32, 2)        # Hidden layers
model_ps = DenseLBDNParams{T}(nu, nh, ny, γ)

function loss(model_ps, x0, qref, uref)
    model = LBDN(model_ps)            # Model
    z = rollout(model, x0, qref)      # Simulation
    return cost(z, qref, uref)        # Cost
end

using Flux

function train_box_ctrl!(
    model_ps, loss_func; 
    epochs=5, lr=1e-3
)
    costs = Vector{Float64}()
    opt_state = Flux.setup(Adam(lr), model_ps)
    for k in 1:epochs

        tloss, dJ = Flux.withgradient(
            loss_func, model_ps, x0, qref, uref)
        Flux.update!(opt_state, model_ps, dJ[1])
        push!(costs, tloss)
    end
    return costs
end

costs = train_box_ctrl!(model_ps, loss)

model   = LBDN(model_ps)
x0_test = zeros(2,60)
qr_test = 2*rand(1, 60) .- 1
z_test  = rollout(model, x0_test, qr_test)

loss2(model, x0, qref, uref) = 
    cost(rollout(model, x0, qref), qref, uref)

function lbdn_compute_times(n; epochs=100)

    # Build model params and a model
    lbdn_ps = DenseLBDNParams{T}(nu, [n], ny, γ)
    diff_lbdn = DiffLBDN(deepcopy(lbdn_ps))

    # Time with LBDN vs DiffLBDN (respectively)
    t_lbdn = @elapsed (
        train_box_ctrl!(lbdn_ps, loss; epochs))
    t_diff_lbdn = @elapsed (
        train_box_ctrl!(diff_lbdn, loss2; epochs))
    return [t_lbdn, t_diff_lbdn]

end

# Evaluate computation time
# Run it once first for just-in-time compiler
ns = 2 .^ (1:3)
lbdn_compute_times(2; epochs=1)
comp_times = reduce(hcat, lbdn_compute_times.(ns))