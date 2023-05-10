mutable struct WrapREN{T} <: AbstractREN{T}
    nl
    nu::Int
    nx::Int
    nv::Int
    ny::Int
    explicit::ExplicitRENParams
    params::AbstractRENParams
end

"""
    WrapREN(ps::AbstractRENParams{T}) where T

Construct REN wrapper from its direct parameterisation.

`WrapREN` is an alternative to [`REN`](@ref) that stores the [`AbstractRENParams`](@ref) and [`ExplicitRENParams`](@ref) within the same object. This means that a new `REN` object does not have to be created each time the parameters are updated. Explicit REN parameters must be updated by the user if the direct parameters have changed.

Note that `WrapREN` cannot be used with [`Flux.jl`](http://fluxml.ai/Flux.jl/stable/), since it relies on mutating the `WrapREN` instance.

# Examples

In this example, we create a REN satisfying some generic behavioural constraints and demonstrate how to update the REN wrapper if model parameters are changed.

```jldoctest; output = false
using LinearAlgebra
using Random
using RobustNeuralNetworks

# Setup
rng = MersenneTwister(42)
batches = 10
nu, nx, nv, ny = 4, 10, 20, 2

Q = Matrix{Float64}(-I(ny))
R = 0.1^2 * Matrix{Float64}(I(nu))
S = zeros(Float64, nu, ny)

# Construct a REN
ren_ps = GeneralRENParams{Float64}(nu, nx, nv, ny, Q, S, R; rng=rng)
ren = WrapREN(ren_ps)

# Some dummy inputs
x0 = init_states(ren, batches; rng=rng)
u0 = randn(rng, ren.nu, batches)

# Evaluate the REN over one timestep
x1, y1 = ren(x0, u0) 

# Update the model after changing a parameter
ren.params.direct.B2 .*= rand(rng, size(ren.params.direct.B2)...)
update_explicit!(ren)

println(round(ren.explicit.B2[1];digits=4))

# output

-0.0335
```

See also [`AbstractREN`](@ref), [`REN`](@ref), and [`DiffREN`](@ref).
"""
function WrapREN(ps::AbstractRENParams{T}) where T
    explicit = direct_to_explicit(ps)
    return WrapREN{T}(ps.nl, ps.nu, ps.nx, ps.nv, ps.ny, explicit, ps)
end

"""
    update_explicit!(m::WrapREN)

Update explicit model in `WrapREN` using the current direct parameters.
"""
function update_explicit!(m::WrapREN)
    m.explicit = direct_to_explicit(m.params)
    return nothing
end

Flux.trainable(m::WrapREN) = Flux.params(m.params)
