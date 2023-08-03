# This file is a part of RobustNeuralNetworks.jl. License is MIT: https://github.com/acfr/RobustNeuralNetworks.jl/blob/main/LICENSE 

mutable struct ContractingRENParams{T} <: AbstractRENParams{T}
    nl::Function                # Sector-bounded nonlinearity
    nu::Int
    nx::Int
    nv::Int
    ny::Int
    direct::DirectRENParams{T}
    αbar::T
end

"""
    ContractingRENParams{T}(nu, nx, nv, ny; <keyword arguments>) where T

Construct direct parameterisation of a contracting REN.

The parameters can be used to construct an explicit [`REN`](@ref) model that has guaranteed, built-in contraction properties.

# Arguments
- `nu::Int`: Number of inputs.
- `nx::Int`: Number of states.
- `nv::Int`: Number of neurons.
- `ny::Int`: Number of outputs.

# Keyword arguments

- `nl::Function=Flux.relu`: Sector-bounded static nonlinearity.

- `αbar::T=1`: Upper bound on the contraction rate with `ᾱ ∈ (0,1]`.

See [`DirectRENParams`](@ref) for documentation of keyword arguments `init`, `ϵ`, `bx_scale`, `bv_scale`, `polar_param`, `D22_zero`, `output_map`, `rng`.

See also [`GeneralRENParams`](@ref), [`LipschitzRENParams`](@ref), [`PassiveRENParams`](@ref).
"""
function ContractingRENParams{T}(
    nu::Int, nx::Int, nv::Int, ny::Int;
    nl::Function        = Flux.relu, 
    αbar::T             = T(1),
    init                = :random,
    polar_param::Bool   = true,
    D22_zero::Bool      = false,
    bx_scale::T         = T(0), 
    bv_scale::T         = T(1), 
    output_map::Bool    = true,
    ϵ::T                = T(1e-12), 
    rng::AbstractRNG    = Random.GLOBAL_RNG
) where T

    # Direct (implicit) params
    direct_ps = DirectRENParams{T}(
        nu, nx, nv, ny; 
        init, ϵ, bx_scale, bv_scale, polar_param, 
        D22_free=true, D22_zero, output_map, rng,
    )

    return ContractingRENParams{T}(nl, nu, nx, nv, ny, direct_ps, αbar)

end

Flux.@functor ContractingRENParams (direct, )

function direct_to_explicit(ps::ContractingRENParams{T}, return_h::Bool=false) where T

    ϵ = ps.direct.ϵ
    ρ = ps.direct.ρ[1]
    X = ps.direct.X
    polar_param = ps.direct.polar_param
    H = x_to_h(X, ϵ, polar_param, ρ)
    
    !return_h && (return hmatrix_to_explicit(ps, H))
    return H

end