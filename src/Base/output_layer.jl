"""
$(TYPEDEF)

Linear output layer of the REN
"""
mutable struct OutputLayer{T}
    C2::Union{Matrix{T},CuMatrix{T}}
    D21::Union{Matrix{T},CuMatrix{T}}
    D22::Union{Matrix{T},CuMatrix{T}}
    by::Union{Vector{T},CuVector{T}}
    D22_trainable::Bool
end

"""
    OutputLayer{T}(nu, nx, nv, ny; ; D22_trainable=false rng=Random.GLOBAL_RNG) where T

Main constructor for the output layer. Randomly generates all matrices
from the Glorot normal distribution, except D22 = 0. Must specify if you
want D22 to be a trainable parameter.
"""
function OutputLayer{T}(nu::Int, nx::Int, nv::Int, ny::Int; D22_trainable=false, rng=Random.GLOBAL_RNG) where T
    C2  = glorot_normal(ny,nx; rng=rng)
    D21 = glorot_normal(ny,nv; rng=rng)
    D22 = zeros(T, ny, nu)                          # TODO: Keep as zeros or initialise as random?
    by  = glorot_normal(ny; rng=rng)
    return OutputLayer{T}(C2, D21, D22, by, D22_trainable)
end

"""
    Flux.trainable(layer::OutputLayer)

Define trainable parameters for `OutputLayer` type
"""
function Flux.trainable(layer::OutputLayer)
    if layer.D22_trainable
        return (layer.C2, layer.D21, layer.D22, layer.by)
    end
    return (layer.C2, layer.D21, layer.by)
end

"""
    Flux.gpu(layer::OutputLayer{T}) where T

Add GPU compatibility for `OutputLayer` type
"""
function Flux.gpu(layer::OutputLayer{T}) where T
    if T != Float32
        println("Moving type: ", T, " to gpu may not be supported. Try Float32!")
    end
    return OutputLayer{T}(gpu(layer.C2), gpu(layer.D21), gpu(layer.D22), gpu(layer.by))
end

"""
    Flux.cpu(layer::OutputLayer{T}) where T

Add CPU compatibility for `OutputLayer` type
"""
function Flux.cpu(layer::OutputLayer{T}) where T
    return OutputLayer{T}(cpu(layer.C2), cpu(layer.D21), cpu(layer.D22), cpu(layer.by))
end
