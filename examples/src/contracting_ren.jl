# This file is a part of RobustNeuralNetworks.jl. License is MIT: https://github.com/acfr/RobustNeuralNetworks.jl/blob/main/LICENSE 

cd(@__DIR__)
using Pkg
Pkg.activate("../")

using CairoMakie
using Random
using RobustNeuralNetworks

rng = MersenneTwister(42)

# Create a contracting REN with just its state as an output 
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
    return ts, y1, y2
end
ts, y1, y2 = simulate()

# Plot trajectories
f1 = Figure(resolution = (500, 300))
ax = Axis(f1[1,1], xlabel="Time samples", ylabel="Internal state",
          title="Contracting RENs forget initial conditions")

lines!(ax, y1, label="Initial condition 1")
lines!(ax, y2, label="Initial condition 2")
axislegend(ax, position=:rb)
display(f1)
save("../../docs/src/assets/contracting_ren.svg", f1)

# Create an animation
p1 = Observable(Point2f[(ts[1], y1[1])])
p2 = Observable(Point2f[(ts[1], y2[1])])

fig = Figure(resolution = 2 .*(600, 400), fontsize=36)
ax = Axis(fig[1,1], xlabel="Time samples", ylabel="Internal state",
          title="Contracting systems forget initial conditions")
scatter!(ax, p1, color="blue", markersize=10)
scatter!(ax, p2, color="orange", markersize=10)
limits!(ax, 0, 600, -16.5, 12.5)

time = 6
framerate = 30
dframe = Int(floor(length(ts) / time / framerate))
frames = 1:dframe:length(ts)

record(fig, "../results/contraction_animation.gif", 1:length(frames); framerate) do i
    if i > 1
        indx = frames[(i-1)]:frames[i]
        new_point1 = Point2f.(ts[indx], y1[indx])
        new_point2 = Point2f.(ts[indx], y2[indx])
        p1[] = append!(p1[], new_point1)
        p2[] = append!(p2[], new_point2)
    end
end
display(fig)
