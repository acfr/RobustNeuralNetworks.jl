# TODO: All of this is outdated. Update it with the parameterisation from
# TODO: Ray's most recent LBDN paper (https://arxiv.org/abs/2301.11526)

mutable struct LBFN{T} <: AbstractLBDN
    As::Tuple
    Bs::Tuple
    bs::Tuple
    ds::Tuple
    γ::T
    nl::Union{typeof(relu), typeof(tanh)}
end

"""
    LBFN{T}(nu, nh, ny, γ; ...)

Constructor for an LBFN with nu inputs, nv outputs, and
`nh = [nh1, nh2,...]` specifying the size of hidden layers.
User-imposed Lipschitz bound `γ` has a default of 1.
"""
function LBFN{T}(
    nu::Int, 
    nh::Vector{Int}, 
    ny::Int,
    γ::T = T(1),
    init = Flux.glorot_uniform, 
    nl = Flux.relu,
    rng = Random.GLOBAL_RNG
) where T

    # Layyer sizes
    lw = [nu, nh..., ny]
    L = length(lw)

    # Initialise params
    As = fill(zeros(T,0,0), L-1)
    Bs = fill(zeros(T,0,0), L-1)
    bs = fill(zeros(T,0), L-1)
    ds = fill(zeros(T,0), L-2)

    # Fill params
    for l in 1:L-1
        As[l] = init(rng, lw[l], lw[l+1] )
        Bs[l] = init(rng, lw[l+1], lw[l+1] )
        bs[l] = 1e-5*init(rng, lw[l+1] )
        (l<L-1) && (ds[l] = init(rng, lw[l+1] ))
    end

    # Return params in tuple (immutable, nicer with Flux)
    return LBFN{T}(tuple(As...), tuple(Bs...), tuple(bs...), tuple(ds...), γ, nl)
end

# Not exactly sure what this does...?
Flux.@functor LBFN

Flux.trainable(m::LBFN) = (m.As, m.Bs, m.bs, m.ds)

"""
    (m::LBFN)(u)

Evaluate an LBFN given some input u.
"""
function (m::LBFN)(u::Union{T,AbstractVecOrMat{T}}) where T

    # Set up
    L = length(m.As)
    y = m.γ .* u
    D = 1
    r2 = T(sqrt(2))

    # Loop the layers
    for l in 1:L
        if l < L

            Γ = exp.( m.ds[l]/2 )
            X = m.As[l]' * m.As[l] + m.Bs[l] - m.Bs[l]'
            W = - 2*r2 * ( ( (I + X) \ m.As[l]' ) ./ Γ ) * D'
            y = m.nl.( W * y .+ m.bs[l] ) 
            D = r2 * Γ .* ((I+X) \ (I-X))

        else
            
            X = m.As[l]' * m.As[l] + m.Bs[l] - m.Bs[l]'
            W = - 2 * ( (I + X) \ m.As[l]' )  * D'
            y = W * y .+ m.bs[l]
            
        end
    end
    return  y
end
