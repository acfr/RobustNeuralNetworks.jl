# Some documentation here...
# Note: Faster to work with tuples than vec of vecs
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

# Constructor for direct params
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