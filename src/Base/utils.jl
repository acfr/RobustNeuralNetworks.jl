# This file is a part of RobustNeuralNetworks.jl. License is MIT: https://github.com/acfr/RobustNeuralNetworks.jl/blob/main/LICENSE 

"""
    glorot_normal(n::Int, m::Int; T=Float64, rng=Random.GLOBAL_RNG)

Generate matrices or vectors from the Glorot normal distribution.
"""
glorot_normal(n::Int, m::Int; T=Float64, rng=Random.GLOBAL_RNG) = 
    convert.(T, glorot_normal(rng, n, m))
glorot_normal(n::Int; T=Float64, rng=Random.GLOBAL_RNG) = 
    convert.(T, glorot_normal(rng, n))


##################################################################
# Utils from Flux.jl
# Copied here to avoid having to load Flux.jl as a package for
# just a few useful functions.

# From: https://github.com/FluxML/Flux.jl/blob/c5650522ffceb9e5a4a03b4c65de9f82b89e68b1/src/utils.jl
nfan() = 1, 1
nfan(n) = 1, n
nfan(n_out, n_in) = n_in, n_out
nfan(dims::Tuple) = nfan(dims...)
nfan(dims...) = prod(dims[1:end-2]) .* (dims[end-1], dims[end])

"""
    glorot_normal([rng], size...; gain = 1) -> Array
    glorot_normal([rng]; kw...) -> Function

From: https://github.com/FluxML/Flux.jl/blob/c5650522ffceb9e5a4a03b4c65de9f82b89e68b1/src/utils.jl

Return an `Array{Float32}` of the given `size` containing random numbers drawn from a normal
distribution with standard deviation `gain * sqrt(2 / (fan_in + fan_out))`,
using [`nfan`](@ref Flux.nfan).

This method is described in [1] and also known as Xavier initialization.

# Examples
```jldoctest; setup = :(using Random; Random.seed!(0))
julia> using Statistics

julia> round(std(Flux.glorot_normal(10, 1000)), digits=3)
0.044f0

julia> round(std(Flux.glorot_normal(1000, 10)), digits=3)
0.044f0

julia> round(std(Flux.glorot_normal(1000, 1000)), digits=3)
0.032f0

julia> Dense(10 => 1000, tanh; init = Flux.glorot_normal(gain=100))
Dense(10 => 1000, tanh)  # 11_000 parameters

julia> round(std(ans.weight), sigdigits=3)
4.45f0
```

# References

[1] Glorot, Xavier, and Yoshua Bengio. "Understanding the difficulty of training deep feedforward neural networks." _Proceedings of the thirteenth international conference on artificial intelligence and statistics_. 2010.
"""
function glorot_normal(rng::AbstractRNG, dims::Integer...; gain::Real=1)
  std = Float32(gain) * sqrt(2.0f0 / sum(nfan(dims...)))
  randn(rng, Float32, dims...) .* std
end
glorot_normal(dims::Integer...; kwargs...) = glorot_normal(default_rng(), dims...; kwargs...)
glorot_normal(rng::AbstractRNG=default_rng(); init_kwargs...) = (dims...; kwargs...) -> glorot_normal(rng, dims...; init_kwargs..., kwargs...)

@non_differentiable glorot_normal(::Any...)
