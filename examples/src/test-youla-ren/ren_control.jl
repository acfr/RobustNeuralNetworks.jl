cd(@__DIR__)
using Pkg
Pkg.activate("../../")

# using Distributions
using LinearAlgebra
using ControlSystems
using JuMP
using Convex
using BSON: @save, @load

using MosekTools
using Mosek

using Plots
using StatsPlots

using Revise

do_plot = false

stop_here

gr()
default(labelfontsize=7, legendfontsize=7, tickfontsize=7,
        linewidth=2, framestyle=:box, label=nothing, grid=true)

# includet("utils.jl")
includet("./models/ffREN.jl")


# First experiment
η = 1E-3
nx = 10  # state dim
nv = 20  # neurons
nu = 1

# Sample simple discrete time system
# Pole locations in polar coordinates
ρ = 0.80
ϕ = 2 * pi / 10
λ = [ρ * cos(ϕ) + ρ * sin(ϕ) * im, ρ * cos(ϕ) - ρ * sin(ϕ) * im]

# Create discrete-time system with 1.0s sampling time
sys = zpk([], λ, 0.3, 1.0)

# Functions to simulate system
Pdy(d) = lsim(sys, d, 1:size(d, 2))[1]
Puy(d) = lsim(sys, d, 1:size(d, 2))[1]

# Sample disurbance and simulate response to that distribution
amp = 10
samples, hold = 30, 50
function sample_disturbance()
    d = 2 * amp * (rand(samples, 1) .- 0.5)
    d = kron(d, ones(hold, 1))
    return d'
end
d = sample_disturbance()

p1 = plot(d[1:1000]; label="Example Disturbance", xlabel="Time Steps")
do_plot && display(p1)

# Simulate system response with disturbance
b = Pdy(d)
p2 = plot(d')
plot!(b')
do_plot && display(p2)




# Sample ESN and simulate effect of output 
Q_param = sample_ff_ren(nu, nx, nv)
function Q(v)
    x0 = init_state(Q_param, size(v, 2))
    xt, wt = simulate(Q_param, x0, [v])

    X = reduce(hcat, xt)
    W = reduce(hcat, wt)

    println(size(X))
    println(size(W))
    
    return [X; W; v; ones(1, size(v, 2))]   # include ones for bias term
end

C = Q(-b)
A = reduce(hcat, Puy(ci') for ci in eachslice(C, dims=1))

stop_here
# Optimize!
θv = Convex.Variable(size(A, 2))

y = A * θv + b
u = C * θv # TODO:

J = norm(y, 1) + η * norm(θv, 2) + 1E-3*sumsquares(u)
constraints = [u < 5.0, u > -5.0]

problem = minimize(J, constraints)
Convex.solve!(problem, Mosek.Optimizer)

θ = evaluate(θv)

# Test input
amps = range(0, length=7, stop=8)
d_test = reduce(hcat, a .* [ones(1, 50) zeros(1, 50)] for a in amps)'

btest = Pdy(d_test)
Ctest = Q(-btest)
Atest = reduce(hcat, Puy(ci) for ci in eachslice(Ctest, dims=2))

# Resulting signals
ytest = Atest * θ + btest
utest = Ctest * θ

# --------------------------- Plotting ----------------------------------------------------


T0 = 25  # washout period

plot(d_test[T0:end]; label="Disturbance")
plot!(Pdy(d_test)[T0:end]; label="Open Loop")
plot!(ytest[T0:end]; label="aREN")
plot!(;xlabel="Time Steps")
p2 = plot!()

# Control input figure

p3 = plot(utest[T0:end]; label="aREN")
plot!([0, length(utest) - T0], [-5, -5]; label="Contraints", c=:black, ls=:dash)
plot!(;xlabel="Time Steps")


plot(p1, p2, p3; layout=(3, 1))
