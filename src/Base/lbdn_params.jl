"""
    mutable struct ExplicitLBDNParams{T, N, M}

Explicit LBDN parameter struct.

These parameters define the explicit form of a Lipschitz-bounded deep network used for model evaluation. Parameters are stored in `NTuple`s, where each element of an `NTuple` is the parameter for a single layer of the network. Tuples are faster to work with than vectors of arrays.

See [Wang et al. (2023)](https://doi.org/10.48550/arXiv.2301.11526) for more details on explicit parameterisations of LBDN.
"""
mutable struct ExplicitLBDNParams{T, N, M}
    A_T::NTuple{N, AbstractMatrix{T}}    # A^T in the paper
    B  ::NTuple{N, AbstractMatrix{T}}
    Ψd ::NTuple{M, AbstractVector{T}}    # Diagonal of matrix Ψ from the paper
    b  ::NTuple{N, AbstractVector{T}}
end

mutable struct DirectLBDNParams{T, N, M}
    XY::NTuple{N, AbstractMatrix{T}}    # [X; Y] in the paper
    α ::NTuple{N, AbstractVector{T}}    # Polar parameterisation
    d ::NTuple{M, AbstractVector{T}}
    b ::NTuple{N, AbstractVector{T}}
end

"""
    DirectLBDNParams{T}(nu, nh, ny; <keyword arguments>) where T

Construct direct parameterisation for a Lipschitz-bounded deep network.

This is typically used by a higher-level constructor to define an LBDN model, which takes the direct parameterisation in `DirectLBDNParams` and defines rules for converting it to an explicit parameterisation. See for example [`DenseLBDNParams`](@ref).

# Arguments

- `nu::Int`: Number of inputs.
- `nh::Vector{Int}`: Number of hidden units for each layer. Eg: `nh = [5,10]` for 2 hidden layers with 5 and 10 nodes (respectively).
- `ny::Int`: Number of outputs.

# Keyword arguments

- `initW::Function=Flux.glorot_normal`: Initialisation function for implicit params `X,Y,d`.

- `initb::Function=Flux.glorot_normal`: Initialisation function for bias vectors.

- `rng::AbstractRNG = Random.GLOBAL_RNG`: rng for model initialisation.

See [Wang et al. (2023)](https://doi.org/10.48550/arXiv.2301.11526) for parameterisation details.

See also [`DenseLBDNParams`](@ref).
"""
function DirectLBDNParams{T}(
    nu::Int, nh::Vector{Int}, ny::Int;
    initW::Function = Flux.glorot_normal,
    initb::Function = Flux.glorot_normal,
    rng::AbstractRNG = Random.GLOBAL_RNG
) where T

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

    return DirectLBDNParams{T,L+1,L}(tuple(XY...), tuple(α...), tuple(d...), tuple(b...))

end

Flux.trainable(m::DirectLBDNParams) = (m.XY, m.α, m.d, m.b)


# TODO: Should add compatibility for layer-wise options
# Eg:
#   - initialisation functions
#   - Activation functions
#   - Whether to include bias or not