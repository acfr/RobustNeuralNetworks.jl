# This file is a part of RobustNeuralNetworks.jl. License is MIT: https://github.com/acfr/RobustNeuralNetworks.jl/blob/main/LICENSE 

using LinearAlgebra
using Random
using RobustNeuralNetworks
using Test

rng = MersenneTwister(42)

"""
Test REN wrapper with General REN params
"""
batches = 20
nu, nx, nv, ny = 4, 5, 10, 2

Q = Matrix{Float64}(-I(ny))
R = 0.1^2 * Matrix{Float64}(I(nu))
S = zeros(Float64, nu, ny)

ren_ps = GeneralRENParams{Float64}(nu, nx, nv, ny, Q, S, R; rng)
ren1 = WrapREN(ren_ps)

x0 = init_states(ren1, batches)
u0 = randn(rng, nu, batches)

# Update the model after changing a parameter
old_B2 = deepcopy(ren1.explicit.B2)
ren1.params.direct.B2 .*= rand(rng, size(ren1.params.direct.B2)...)

x1, y1 = ren1(x0, u0)
update_explicit!(ren1)

new_B2 = deepcopy(ren1.explicit.B2)
@test old_B2 != new_B2