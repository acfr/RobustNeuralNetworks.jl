# This file is a part of RobustNeuralNetworks.jl. License is MIT: https://github.com/acfr/RobustNeuralNetworks.jl/blob/main/LICENSE 

"""
    mutable struct ExplicitLBDNParams{T, N, M}

Explicit LBDN parameter struct.

These parameters define the explicit form of a Lipschitz-bounded deep network used for model evaluation. Parameters are stored in `NTuple`s, where each element of an `NTuple` is the parameter for a single layer of the network. Tuples are faster to work with than vectors of arrays.

See [Wang et al. (2023)](https://proceedings.mlr.press/v202/wang23v.html) for more details on explicit parameterisations of LBDN.
"""
mutable struct ExplicitLBDNParams{T, N, M}
    A_T::NTuple{N, AbstractMatrix{T}}   # A^T in the paper
    B  ::NTuple{N, AbstractMatrix{T}}
    Ψd ::NTuple{M, AbstractVector{T}}   # Diagonal of matrix Ψ from the paper
    b  ::NTuple{N, AbstractVector{T}}
    sqrtγ::AbstractVector{T}
end

# No trainable params
@functor ExplicitLBDNParams
trainable(m::ExplicitLBDNParams) = (; )

mutable struct DirectLBDNParams{T, N, M}
    XY::NTuple{N, AbstractMatrix{T}}    # [X; Y] in the paper
    α ::NTuple{N, AbstractVector{T}}    # Polar parameterisation
    d ::NTuple{M, AbstractVector{T}}
    b ::NTuple{N, AbstractVector{T}}
    log_γ::AbstractVector{T}            # Store ln(γ) so that √exp(logγ) is positive
    learn_γ::Bool
end

"""
    DirectLBDNParams{T}(nu, nh, ny, γ; <keyword arguments>) where T

Construct direct parameterisation for a Lipschitz-bounded deep network.

This is typically used by a higher-level constructor to define an LBDN model, which takes the direct parameterisation in `DirectLBDNParams` and defines rules for converting it to an explicit parameterisation. See for example [`DenseLBDNParams`](@ref).

# Arguments

- `nu::Int`: Number of inputs.
- `nh::Union{Vector{Int}, NTuple{N, Int}}`: Number of hidden units for each layer. Eg: `nh = [5,10]` for 2 hidden layers with 5 and 10 nodes (respectively).
- `ny::Int`: Number of outputs.
- `γ::Real=T(1)`: Lipschitz upper bound, must be positive.

# Keyword arguments

- `initW::Function=Flux.glorot_normal`: Initialisation function for implicit params `X,Y,d`.

- `initb::Function=Flux.glorot_normal`: Initialisation function for bias vectors.

- `learn_γ::Bool=false:` Whether to make the Lipschitz bound γ a learnable parameter.

- `rng::AbstractRNG = Random.GLOBAL_RNG`: rng for model initialisation.

See [Wang et al. (2023)](https://proceedings.mlr.press/v202/wang23v.html) for parameterisation details.

See also [`DenseLBDNParams`](@ref).
"""
function DirectLBDNParams{T}(
    nu::Int, nh::Union{Vector{Int}, NTuple{N, Int}}, 
    ny::Int, γ::Real = T(1);
    initW::Function  = glorot_normal,
    initb::Function  = glorot_normal,
    learn_γ::Bool    = false,
    rng::AbstractRNG = Random.GLOBAL_RNG
) where {T, N}

    n = [nu, nh..., ny]
    L = length(nh)

    XY = fill(zeros(T,0,0), L+1)
    b  = fill(zeros(T,0), L+1)
    α  = fill(zeros(T,0), L+1)
    d  = fill(zeros(T,0), L)

    for k in 1:L+1
        XY[k] = initW(rng, n[k+1] + n[k], n[k+1])
        α[k]  = [norm(XY[k])]
        b[k]  = initb(rng, n[k+1])
        (k<L+1) && (d[k] = initW(rng, n[k+1]))
    end

    log_γ = [T(log(γ))]
    return DirectLBDNParams{T,L+1,L}(
        tuple(XY...), tuple(α...), tuple(d...), tuple(b...), log_γ, learn_γ
    )
end

@functor DirectLBDNParams
function trainable(m::DirectLBDNParams)
    if m.learn_γ
        return (XY=m.XY, α=m.α, d=m.d, b=m.b, log_γ=m.log_γ)
    else 
        return (XY=m.XY, α=m.α, d=m.d, b=m.b)
    end
end

# TODO: Should add compatibility for layer-wise options
# Eg:
#   - initialisation functions
#   - Activation functions
#   - Whether to include bias or not