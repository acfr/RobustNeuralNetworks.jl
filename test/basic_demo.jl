using LinearAlgebra
using Random
using RecurrentEquilibriumNetworks
using Test

"""
Test basic functionality with contracting REN
"""
batches = 50
nu, nx, nv, ny = 4, 10, 20, 2

contracting_ren_ps = ContractingRENParams{Float64}(nu, nx, nv, ny)
contracting_ren = REN(contracting_ren_ps)

x0 = init_states(contracting_ren, batches)
u0 = randn(contracting_ren.nu, batches)

x1, y1 = contracting_ren(x0, u0)  # Evaluates the REN over one timestep

@test typeof(x1) <: Matrix{Float64}


"""
Test REN wrapper with General REN params.
Slightly modified from version in the README (uses Q,S,R)
"""
Q = Matrix{Float64}(-I(ny))
R = 0.1^2 * Matrix{Float64}(I(nu))
S = zeros(Float64, nu, ny)

ren_ps = GeneralRENParams{Float64}(nu, nx, nv, ny, Q, S, R)
ren = WrapREN(ren_ps)

x0 = init_states(ren, batches)
u0 = randn(ren.nu, batches)

x1, y1 = ren(x0, u0)  # Evaluates the REN over one timestep

# Update the model after changing a parameter
old_B2 = deepcopy(ren.explicit.B2)
ren.params.direct.B2 .*= rand(size(ren.params.direct.B2)...)
update_explicit!(ren)
new_B2 = deepcopy(ren.explicit.B2)

@test old_B2 != new_B2