cd(@__DIR__)
using Pkg
Pkg.activate("../")

using ControlSystems
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

# Sample disturbance input
function sample_disturbance(amplitude=10, samples=30, hold=50)
    w = 2 * amplitude * (rand(rng, 1, samples) .- 0.5)
    return kron(w, ones(1, hold))
end

w = sample_disturbance()