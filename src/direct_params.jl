"""
$(TYPEDEF)

Direct (implicit) parameters used to construct a REN.
"""
mutable struct DirectParams{T}
    nl                                  # nonlinearity
    ρ::Union{Vector{T},CuVector{T}}     # used in polar param
    V::Union{Matrix{T},CuMatrix{T}}
    S_1::Union{Matrix{T},CuMatrix{T}}
    B2::Union{Matrix{T},CuMatrix{T}}
    D12::Union{Matrix{T},CuMatrix{T}}
    bx::Union{Vector{T},CuVector{T}}
    bv::Union{Vector{T},CuVector{T}}
    ϵ::T
    polar_param::Bool                   # Whether or not to use polar param
end

"""
Constructor for `DirectParams` struct. Allows for the following
initialisation methods, specified as symbols by `init` argument:
- `:random`: Random sampling for all parameters
- `:cholesky`: Compute `V` with cholesky factorisation of `H`, sets `E,F,P = I` 
"""
function DirectParams{T}(
    nu::Int, nx::Int, nv::Int; 
    init = :random,
    nl = Flux.relu, 
    ϵ = T(0.001), 
    bx_scale = T(0), 
    bv_scale = T(1), 
    polar_param = false,
    rng = Random.GLOBAL_RNG
) where T

    # Random sampling
    if init == :random

        B2  = glorot_normal(nx, nu; T=T, rng=rng)
        D12 = glorot_normal(nv, nu; T=T, rng=rng)
        
        ρ = zeros(1)
        
        # Make orthogonal V
        V = glorot_normal(2nx + nv, 2nx + nv; T=T, rng=rng)
        V = Matrix(qr(V).Q)

        S_1 = glorot_normal(nx, nx; T=T, rng=rng)

        bv = T(bv_scale) * glorot_normal(nv; T=T, rng=rng)
        bx = T(bx_scale) * glorot_normal(nx; T=T, rng=rng)

    # Specify H and compute V
    elseif init == :cholesky

        E = Matrix{T}(I, nx, nx)
        F = Matrix{T}(I, nx, nx)
        P = Matrix{T}(I, nx, nx)

        B1 = zeros(T, nx, nv)
        B2 = glorot_normal(nx, nu; T=T, rng=rng)

        C1  = zeros(T, nv, nx)
        D11 = glorot_normal(nv, nv; T=T, rng=rng)
        D12 = zeros(T, nv, nu)

        # TODO: This is prone to errors. Needs a bugfix!
        Λ = 2*I
        H22 = 2Λ - D11 - D11'
        Htild = [(E + E' - P) -C1' F';
                 -C1 H22 B1'
                 F  B1  P] + ϵ * I
        
        ρ = zeros(T, 1)
        V = Matrix{T}(cholesky(Htild).U) # H = V'*V

        S_1 = glorot_normal(nx, nx; T=T, rng=rng)

        bv = T(bv_scale) * glorot_normal(nv; T=T, rng=rng)
        bx = T(bx_scale) * glorot_normal(nx; T=T, rng=rng)

    else
        error("Undefined initialisation method ", init)
    end

    return DirectParams(nl, ρ ,V, S_1, B2, D12, bx, bv, T(ϵ), polar_param)
end

# Trainable params. Filter empty ones (handy when nx=0)
Flux.trainable(L::DirectParams) = filter(
    p -> length(p) !=0, 
    [L.ρ, L.V, L.S_1, L.B2, L.D12, L.bx, L.bv]
)

# GPU/CPU compatibility
function Flux.gpu(M::DirectParams{T}) where T
    if T != Float32
        println("Moving type: ", T, " to gpu may not be supported. Try Float32!")
    end
    return DirectParams{T}(
        M.ϕ, gpu(M.V), gpu(M.S_1), gpu(M.B2),
        gpu(M.D12), gpu(M.bx), gpu(M.bv), M.ϵ
    )
end

function Flux.cpu(M::DirectParams{T}) where T
    return DirectParams{T}(
        M.ϕ, cpu(M.V), cpu(M.S_1), cpu(M.B2),
        cpu(M.D12), cpu(M.bx), cpu(M.bv), M.ϵ
    )
end
