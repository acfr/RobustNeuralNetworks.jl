# This file is a part of RobustNeuralNetworks.jl. License is MIT: https://github.com/acfr/RobustNeuralNetworks.jl/blob/main/LICENSE 

cd(@__DIR__)
using Pkg
Pkg.activate("../")

using BSON
using CairoMakie
using ControlSystemsBase
using Convex
using LinearAlgebra
using Mosek, MosekTools
using Random
using RobustNeuralNetworks

rng = MersenneTwister(1)

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
function sample_disturbance(amplitude=10, samples=500, hold=50)
    d = 2 * amplitude * (rand(rng, 1, samples) .- 0.5)
    return kron(d, ones(1, hold))
end
d = sample_disturbance()

# Check out the disturbance
f = Figure(resolution = (600, 400))
ax = Axis(f[1,1], xlabel="Time steps", ylabel="Output")
lines!(ax, vec(d)[1:1000],  label="Disturbance")
axislegend(ax, position=:rt)
display(f)
save("../results/echo-ren/echo_ren_inputs.svg", f)

# Set up a contracting REN whose outputs are yt = [xt; wt; ut]
nu = 1
nx, nv = 50, 500
ny = nx + nv + nu
ren_ps = ContractingRENParams{Float64}(nu, nx, nv, ny; rng)
model  = REN(ren_ps)

model.explicit.C2  .= [I(nx); zeros(nv, nx); zeros(nu, nx)]
model.explicit.D21 .= [zeros(nx, nv); I(nv); zeros(nu, nv)]
model.explicit.D22 .= [zeros(nx, nu); zeros(nv, nu); I(nu)]
model.explicit.by  .= zeros(ny)

# Echo-state network params θ = [C2, D21, D22, by]
θ = Convex.Variable(1, nx+nv+nu+1)

# Echo-state components (add ones for bias vector)
function Qᵢ(u)
    x0 = init_states(model, size(u,2))
    _, y = model(x0, u)
    return [y; ones(1,size(y,2))]
end

# Complete the closed-loop response and control inputs 
# z = T₀ + ∑ θᵢ*T₁(Qᵢ(T₂(d)))
# u = ∑ θᵢ*Qᵢ(T₂(d))
function sim_echo_state_network(d, θ)
    z0 = T0(d)
    ỹ  = T2(d)
    ũ  = Qᵢ(ỹ)
    z1 = reduce(vcat, T1(ũ') for ũ in eachrow(ũ))
    z  = z0 + θ * z1
    u  = θ * ũ
    return z, u, z0
end
z, u, _= sim_echo_state_network(d, θ)

# Cost function and constraints
J = norm(z, 1) + 1e-4*(sumsquares(u) + norm(θ, 2))
constraints = [u < 5, u > -5]

# Optimise the closed-loop response
problem = minimize(J, constraints)
Convex.solve!(problem, Mosek.Optimizer)

u1 = evaluate(u)
println("Maximum training controls: ", round(maximum(u1), digits=2))
println("Minimum training controls: ", round(minimum(u1), digits=2))
println("Training cost: ", round(evaluate(J), digits=2), "\n")

# Test on different inputs
θ_solved = evaluate(θ)
a_test = range(0, length=7, stop=8)
d_test = reduce(hcat, a .* [ones(1, 50) zeros(1, 50)] for a in a_test)
z_test, u_test, z0_test = sim_echo_state_network(d_test, θ_solved)

println("Maximum test controls: ", round(maximum(u_test), digits=2))
println("Minimum test controls: ", round(minimum(u_test), digits=2))
bson("../results/echo-ren/echo_ren_params.bson", Dict("params" => θ_solved))

# Plot the results
f = Figure(resolution = (1000, 400))
ga = f[1,1] = GridLayout()

# Response
ax1 = Axis(ga[1,1], xlabel="Time steps", ylabel="Output")
lines!(ax1, vec(d_test),  label="Disturbance")
lines!(ax1, vec(z0_test), label="Open Loop")
lines!(ax1, vec(z_test),  label="Echo-REN")
axislegend(ax1, position=:lt)

# Control inputs
ax2 = Axis(ga[1,2], xlabel="Time steps", ylabel="Control signal")
lines!(ax2, vec(u_test), label="Echo-REN")
lines!(
    ax2, [1, length(u_test)], [-5, -5], 
    color=:black, linestyle=:dash, label="Constraints"
)
lines!(ax2, [1, length(u_test)], [5, 5], color=:black, linestyle=:dash)
axislegend(ax2, position=:rt)

display(f)
save("../results/echo-ren/echo_ren_results.svg", f)