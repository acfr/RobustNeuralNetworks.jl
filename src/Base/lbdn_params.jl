mutable struct DirectLBDNParams{T}
    nl          # Activation function
    X::Tuple    # TODO: Should these be a tuple or another data structure?
    Y::Tuple    # TODO: Can probably combine X and Y
    d::Tuple
    b::Tuple
    γ::T
end

# TODO: Should add compatibility for layer-wise options
# Eg:
#   - initialisation functions
#   - Activation functions
#   - Whether to include bias or not (this should be added!)

# Constructor for direct params
function DirectLBDNParams{T}(
    nu::Int, nh::Vector{Int}, ny::Int, γ::Number;
    nl,
    initW = Flux.glorot_normal,
    initb = Flux.glorot_normal,
    rng = Random.GLOBAL_RNG
) where T

    # Layer sizes
    n = [nu, nh..., ny]
    L = length(lw)

    # Initialise params
    X = fill(zeros(T,0,0), L)
    Y = fill(zeros(T,0,0), L)
    b = fill(zeros(T,0), L)
    d = fill(zeros(T,0), L-1)

    # Fill them out
    for k in 0:L
        X[k] = initW(rng, n[k+1], n[k+1])
        Y[k] = initW(rng, n[k], n[k+1])
        b[k] = initb(rng, n[k+1])
        (k<L) && (d[k] = initW(rng, n[k+1]))
    end

    return DirectLBDNParams{T}(nl, X, Y, d, b, T(γ))

end

Flux.trainable(m::DirectLBDNParams) = (m.X, m.Y, m.d, m.b)
