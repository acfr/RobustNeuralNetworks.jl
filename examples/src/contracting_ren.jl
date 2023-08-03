# This file is a part of RobustNeuralNetworks.jl. License is MIT: https://github.com/acfr/RobustNeuralNetworks.jl/blob/main/LICENSE 

cd(@__DIR__)
using Pkg
Pkg.activate("../")

using CairoMakie
using Random
using RobustNeuralNetworks

rng = MersenneTwister(42)

# Create a contracting REN with just its state as an output, slow dynamics
nu, nx, nv, ny = 1, 1, 10, 1
ren_ps = ContractingRENParams{Float64}(nu, nx, nv, ny; output_map=false, rng, init=:cholesky)
ren = REN(ren_ps)

# Make it converge a little faster...
ren.explicit.A .-= 1e-2

# Simulate it from different initial conditions
function simulate()

    # Different initial conditions
    x1 = 5*randn(rng, nx)
    x2 = -deepcopy(x1)

    # Same inputs
    ts = 1:600
    u = sin.(0.1*ts)

    # Keep track of outputs
    y1 = zeros(length(ts))
    y2 = zeros(length(ts))

    # Simulate and return outputs
    for t in ts
        x1, ya = ren(x1, u[t:t])
        x2, yb = ren(x2, u[t:t])
        y1[t] = ya[1]
        y2[t] = yb[1]
    end
    return y1, y2
end
y1, y2 = simulate()

# Plot trajectories
fig = Figure(resolution = (500, 300))
ax = Axis(fig[1,1], xlabel="Time samples", ylabel="Internal state",
          title="Contracting RENs forget initial conditions")

lines!(ax, y1, label="Initial condition 1")
lines!(ax, y2, label="Initial condition 2")
axislegend(ax, position=:rb)
display(fig)
save("../../docs/src/assets/contracting_ren.svg", fig)