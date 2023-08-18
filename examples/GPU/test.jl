# This file is a part of RobustNeuralNetworks.jl. License is MIT: https://github.com/acfr/RobustNeuralNetworks.jl/blob/main/LICENSE 

cd(@__DIR__)
using Pkg
Pkg.activate("..")

using CUDA
using Flux
using Random
using RobustNeuralNetworks

rng = MersenneTwister(0)
dev = gpu
T = Float32


#####################################################################
# Problem setup

# System parameters
m = 1                   # Mass (kg)
k = 5                   # Spring constant (N/m)
μ = 0.5                 # Viscous damping coefficient (kg/m)
nx = 2                  # Number of states

# Continuous and discrete dynamics and measurements
_visc(v::Matrix) = μ * v .* abs.(v)
f(x::Matrix,u::Matrix) = [x[2:2,:]; (u[1:1,:] - k*x[1:1,:] - _visc(x[2:2,:]))/m]
fd(x,u) = x + dt*f(x,u)
gd(x::Matrix) = x[1:1,:]

# Generate training data
dt = T(0.01)            # Time-step (s)
Tmax = 0.1               # Simulation horizon
ts = 1:Int(round(Tmax/dt))     # Time array indices

batches = 20
u  = fill(zeros(T, 1, batches), length(ts)-1)
X  = fill(zeros(T, 1, batches), length(ts))
X[1] = (2*rand(rng, T, nx, batches) .- 1) / 2

for t in ts[1:end-1]
    X[t+1] = fd(X[t],u[t])
end

Xt = X[1:end-1]
y = gd.(Xt)

# Store data for debugging
observer_data = [[ut; yt] for (ut,yt) in zip(u, y)]
indx = shuffle(rng, 1:length(observer_data))
xt = Xt[indx[1]] |> dev
inputs = observer_data[indx[1]] |> dev



#####################################################################
# Train a model

# Define a REN model for the observer
nv = 20
nu = size(observer_data[1], 1)
ny = nx
model_ps = ContractingRENParams{T}(nu, nx, nv, ny; output_map=false, rng)
model = DiffREN(model_ps) |> dev

function test_me(func, args...)
    out = func(args...)
    all_good = true
    for _ in 1:10000
        out1 = func(args...)
        (out1 != out) && (all_good = false)
        !all_good && (println(out .- out1); break)
        out = out1
    end
    return all_good
end


explicit = direct_to_explicit(model.params) #|> dev
b0 = cu(randn(rng, T, model.nv, size(xt,2)))
b1 = b0 |> cpu

function f_mod(b, D11)
    # wt = RobustNeuralNetworks.tril_eq_layer(tanh, D11, b)
    wt = RobustNeuralNetworks.solve_tril_layer(tanh, D11, b)
    return wt
end

using LinearAlgebra

function f1_mod(b, D11)
    z_eq  = similar(b)
    Di_zi = typeof(b)(zeros(Float32, 1, size(b,2))) # Using similar(b, 1, size(b,2)) induces NaN on GPU
    for i in axes(b,1)
        Di = @view D11[i:i, 1:i - 1]
        zi = @view z_eq[1:i-1,:]
        bi = @view b[i:i, :]

        mul!(Di_zi, Di, zi)
        z_eq[i:i,:] .= σ.(Di_zi .+ bi)  
    end
    return z_eq
end

println("Model call correct? ", test_me(f1_mod, b0, explicit.D11))

f(b) = typeof(b)(zeros(size(b)...))

# function solve_tril_layer(σ::F, D11, b) where F
#     z_eq  = similar(b)
#     Di_zi = similar(z_eq, 1, size(b,2))
#     for i in axes(b,1)
#         Di = @view D11[i:i, 1:i - 1]
#         zi = @view z_eq[1:i-1,:]
#         bi = @view b[i:i, :]

#         mul!(Di_zi, Di, zi)
#         z_eq[i:i,:] .= σ.(Di_zi .+ bi)  
#     end
#     return z_eq
# end



"""
Some observations:

- If I run it with REN it seems to work
- If I run it with DiffREN it does not
- When I run it with DiffREN and then REN in the same Julia session, sometimes the second run with REN fails too.
- This is the case even when running test_me() with 100000 iterations.
- The difference seems to be whether or not I send the explicit params directly to the GPU?
"""

println()