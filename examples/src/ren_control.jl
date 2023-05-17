cd(@__DIR__)
using Pkg
Pkg.activate("../")

using CairoMakie
using ControlSystems
using Convex
using LinearAlgebra
using Random
using RobustNeuralNetworks

do_plot = false
rng = MersenneTwister(42)

# System parameters and poles: λ = ρ*exp(± im ϕ)
ρ = 0.8
ϕ = 0.2π
λ = ρ .* [cos(ϕ) + sin(ϕ)*im, cos(ϕ) - sin(ϕ)*im] #exp.(im*ϕ.*[1,-1])

# Construct discrete-time system with gain 0.3, sampling time 1.0s
k = 0.3
Ts = 1.0
sys = zpk([], λ, k, Ts)

# Closed-loop system components
sim_sys(u::AbstractMatrix) = lsim(sys, u, 1:size(u,2))[1]
T0(u) = sim_sys(u)
T1(u) = sim_sys(u)
T2(u) = -sim_sys(u)

# Sample disturbances
function sample_disturbance(amplitude=10, samples=30, hold=50)
    d = 2 * amplitude * (rand(rng, 1, samples) .- 0.5)
    return kron(d, ones(1, hold))
end
d = sample_disturbance()
batches = size(d, 2)

# Compute responses
z0 = T0(d)
ỹ = T2(d)

# Plot just to check
if do_plot
    f = lines(vec(d))
    lines!(vec(z))
    display(f)
end

# Echo-state network: 
# Set up a contracting REN whose outputs are yt = [xt; wt; ut]
nu = 1
nx, nv = 10, 20
ny = nx + nv + nu
ren_ps = ContractingRENParams{Float64}(nu, nx, nv, nx; rng=rng)
model  = REN(ren_ps)

model.explicit.C2  = Matrix{Float64}([I(nx); zeros(nv, nx); zeros(nu, nx)])
model.explicit.D21 = Matrix{Float64}([zeros(nx, nv); I(nv); zeros(nu, nv)])
model.explicit.D22 = Matrix{Float64}([zeros(nx, nu); zeros(nv, nu); I(nu)])
model.explicit.by  = zeros(ny)

# Set up a variable to represent the actual output layer
θ = Convex.Variable(1, nx+nv+nu+1)

# Compute closed-loop response
function Q(ỹt)
    x0 = init_states(model, batches)
    _, yt = model(x0, ỹt)
    ũ = θ * [yt; ones(1, batches)] # Add 1s for the bias vector
    return ũ
end
ũ = Q(ỹ)
z = z0 + T1(ũ)

# TODO: The output layer is linear. You can take it outside of the next operation! (will fix bug)