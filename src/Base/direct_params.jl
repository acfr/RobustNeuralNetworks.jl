"""
$(TYPEDEF)

Direct (implicit) parameters used to construct a REN.
"""
mutable struct DirectParams{T}
    nl                                  # nonlinearity
    ρ::Union{Vector{T},CuVector{T}}     # used in polar param
    V::Union{Matrix{T},CuMatrix{T}}
    Y1::Union{Matrix{T},CuMatrix{T}}
    X3::Union{Matrix{T},CuMatrix{T}}
    Y3::Union{Matrix{T},CuMatrix{T}}
    Z3::Union{Matrix{T},CuMatrix{T}}
    B2::Union{Matrix{T},CuMatrix{T}}
    D12::Union{Matrix{T},CuMatrix{T}}
    bx::Union{Vector{T},CuVector{T}}
    bv::Union{Vector{T},CuVector{T}}
    ϵ::T
    polar_param::Bool                   # Whether or not to use polar param
    D22_free::Bool                      # Is D22 free or parameterised by (X3,Y3,Z3)?
end

"""
    DirectParams{T}(nu, nx, nv; ...)

Constructor for `DirectParams` struct. Allows for the following
initialisation methods, specified as symbols by `init` argument:
- `:random`: Random sampling for all parameters
- `:cholesky`: Compute `V` with cholesky factorisation of `H`, sets `E,F,P = I`

Option `D22_free` specifies whether or not to include parameters X3, Y3, and Z3
as trainable parameters used in the explicit construction of D22. If `D22_free == true`
then `(X3,Y3,Z3)` are empty and not trainable.
Note that `D22_free = false` by default.
"""
function DirectParams{T}(
    nu::Int, nx::Int, nv::Int, ny::Int; 
    init = :random,
    nl = Flux.relu, 
    ϵ = T(0.001), 
    bx_scale = T(0), 
    bv_scale = T(1), 
    polar_param = false,
    D22_free = false,
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

    else
        error("Undefined initialisation method ", init)
    end

    # Free parameter for E
    Y1 = glorot_normal(nx, nx; T=T, rng=rng)

    # Parameters for D22 in output layer
    if D22_free
        X3 = zeros(T, 0, 0)
        Y3 = zeros(T, 0, 0)
        Z3 = zeros(T, 0, 0)
    else
        X3 = glorot_normal(nu, nu; T=T, rng=rng)
        Y3 = glorot_normal(nu, nu; T=T, rng=rng)
        Z3 = glorot_normal(abs(ny - nu), ny;  T=T, rng=rng)
    end

    # Bias terms
    bv = T(bv_scale) * glorot_normal(nv; T=T, rng=rng)
    bx = T(bx_scale) * glorot_normal(nx; T=T, rng=rng)

    return DirectParams(
        nl, ρ ,V, 
        Y1, X3, Y3, Z3, 
        B2, D12, bx, bv, T(ϵ), 
        polar_param, D22_free
)
end

"""
    Flux.trainable(L::DirectParams)

Define trainable parameters for `DirectParams` type.
Filter empty ones (handy when nx=0)
"""
function Flux.trainable(L::DirectParams)
    if L.D22_free
        return filter(
            p -> length(p) !=0, 
            [L.ρ, L.V, L.Y1, L.B2, L.D12, L.bx, L.bv]
        )
    end
    return filter(
        p -> length(p) !=0, 
        [L.ρ, L.V, L.Y1, L.X3, L.Y3, L.Z3, L.B2, L.D12, L.bx, L.bv]
    )
end

"""
    Flux.gpu(M::DirectParams{T}) where T

Add GPU compatibility for `DirectParams` type
"""
function Flux.gpu(M::DirectParams{T}) where T
    if T != Float32
        println("Moving type: ", T, " to gpu may not be supported. Try Float32!")
    end
    return DirectParams{T}(
        M.ϕ, gpu(M.V), gpu(M.Y1), gpu(M.X3), gpu(M.Y3), 
        gpu(M.Z3), gpu(M.B2), gpu(M.D12), gpu(M.bx), 
        gpu(M.bv), M.ϵ, M.polar_param
    )
end

"""
    Flux.cpu(M::DirectParams{T}) where T

Add CPU compatibility for `DirectParams` type
"""
function Flux.cpu(M::DirectParams{T}) where T
    return DirectParams{T}(
        M.ϕ, cpu(M.V), cpu(M.Y1), cpu(M.X3), cpu(M.Y3), 
        cpu(M.Z3), cpu(M.B2), cpu(M.D12), cpu(M.bx), 
        cpu(M.bv), M.ϵ, M.polar_param
    )
end
