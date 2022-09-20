"""
$(TYPEDEF)

Linear output layer of the REN
"""
mutable struct Output{T}
    C2::Union{Matrix{T},CuMatrix{T}}
    D21::Union{Matrix{T},CuMatrix{T}}
    D22::Union{Matrix{T},CuMatrix{T}}
    by::Union{Vector{T},CuVector{T}}
end
function Output{T}(nu::Int, nx::Int, nv::Int, ny::Int; rng=Random.GLOBAL_RNG) where T
    C2  = glorot_normal(ny,nx; rng=rng)
    D21 = glorot_normal(ny,nv; rng=rng)
    D22 = zeros(T, ny, nu)                          # TODO: Why zeros?
    by  = glorot_normal(ny; rng=rng)
    return Output{T}(C2, D21, D22, by)
end

# Trainable params
Flux.trainable(layer::Output) = (layer.C2, layer.D21, layer.D22, layer.by)

# Call output layer
(layer::Output)(x, w, u) = layer.C2 * x + layer.D21 * w + layer.D22 * u .+ layer.by

# GPU/CPU compatibility
function Flux.gpu(layer::Output{T}) where T
    if T != Float32
        println("Moving type: ", T, " to gpu may not be supported. Try Float32!")
    end
    return Output{T}(gpu(layer.C2), gpu(layer.D21), gpu(layer.D22), gpu(layer.by))
end

function Flux.cpu(layer::Output{T}) where T
    return Output{T}(cpu(layer.C2), cpu(layer.D21), cpu(layer.D22), cpu(layer.by))
end
