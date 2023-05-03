# Some documentation here...
# TODO: Are tuples faster than vectors here?
mutable struct ExplicitLBDNParams{T}
    A::Vector{Matrix{T}}    # A^T in the paper
    B::Vector{Matrix{T}}
    Ψ::Vector{Vector{T}}    # Just the diagonal component of Ψ from the paper
    b::Vector{Vector{T}}
end

mutable struct DirectLBDNParams{T, N, M}
    X::NTuple{N, Matrix{T}}
    Y::NTuple{N, Matrix{T}}    # TODO: Can probably combine X and Y
    d::NTuple{N, Vector{T}}
    b::NTuple{M, Vector{T}}
end

# TODO: Should add compatibility for layer-wise options
# Eg:
#   - initialisation functions
#   - Activation functions
#   - Whether to include bias or not (this should be added!)

# Constructor for direct params
function DirectLBDNParams{T}(
    nu::Int, nh::Vector{Int}, ny::Int;
    initW::Function = Flux.glorot_normal,
    initb::Function = Flux.glorot_normal,
    rng::AbstractRNG = Random.GLOBAL_RNG
) where T

    # Layer sizes
    n = [nu, nh..., ny]
    L = length(nh)

    # Initialise params
    X = fill(zeros(T,0,0), L+1)
    Y = fill(zeros(T,0,0), L+1)
    b = fill(zeros(T,0), L+1)
    d = fill(zeros(T,0), L)

    # Fill them in
    for k in 1:L+1
        X[k] = initW(rng, n[k+1], n[k+1])
        Y[k] = initW(rng, n[k], n[k+1])
        b[k] = initb(rng, n[k+1])
        (k<L+1) && (d[k] = initW(rng, n[k+1]))
    end

    # TODO: Nicer way of creating tuples here would be nice
    return DirectLBDNParams{T,L+1,L}(tuple(X...), tuple(Y...), tuple(d...), tuple(b...))

end

Flux.trainable(m::DirectLBDNParams) = (m.X, m.Y, m.d, m.b)
