mutable struct DirectParams{T}
    ρ::Union{Vector{T},CuVector{T}}     # used in polar param
    X::Union{Matrix{T},CuMatrix{T}}
    Y1::Union{Matrix{T},CuMatrix{T}}
    X3::Union{Matrix{T},CuMatrix{T}}
    Y3::Union{Matrix{T},CuMatrix{T}}
    Z3::Union{Matrix{T},CuMatrix{T}}
    B2::Union{Matrix{T},CuMatrix{T}}
    C2::Union{Matrix{T},CuMatrix{T}}
    D12::Union{Matrix{T},CuMatrix{T}}
    D21::Union{Matrix{T},CuMatrix{T}}
    D22::Union{Matrix{T},CuMatrix{T}}
    bx::Union{Vector{T},CuVector{T}}
    bv::Union{Vector{T},CuVector{T}}
    by::Union{Vector{T},CuVector{T}}
    ϵ::T
    polar_param::Bool                   # Whether or not to use polar param
    D22_free::Bool                      # Is D22 free or parameterised by (X3,Y3,Z3)?
    D22_zero::Bool                      # Option to remove feedthrough.
end

"""
    DirectParams{T}(nu, nx, nv; <keyword arguments>) where T

Construct direct parameterisation for an (acyclic) recurrent equilibrium network.

This is typically used by higher-level constructors when defining a REN, which take the direct parameterisation and define rules for converting it to an explicit parameterisation. See for example [`GeneralRENParams`](@ref).
    
# Arguments

- `nu::Int`: Number of inputs.
- `nx::Int`: Number of states.
- `nv::Int`: Number of neurons.
    
# Keyword arguments

- `init=:random`: Initialisation method. Options are:
    - `:random`: Random sampling for all parameters.
    - `:cholesky`: Compute `X` with cholesky factorisation of `H`, sets `E,F,P = I`.

- `polar_param::Bool=true`: Use polar parameterisation to construct `H` matrix from `X` in REN parameterisation (recommended).

- `D22_free::Bool=false`: Specify whether to train `D22` as a free parameter (`true`), or construct it separately from `X3, Y3, Z3` (`false`). Typically use `D22_free = true` only for a contracting REN.

- `D22_zero::Bool=false`: Fix `D22 = 0` to remove any feedthrough.

- `bx_scale::T=0`: Set scale of initial state bias vector `bx`.

- `bv_scale::T=1`: Set scalse of initial neuron input bias vector `bv`.

- `ϵ::T=1e-12`: Regularising parameter for positive-definite matrices.

- `rng::AbstractRNG=Random.GLOBAL_RNG`: rng for model initialisation.

See [Revay et al. (2021)](https://arxiv.org/abs/2104.05942) for parameterisation details.

See also [`GeneralRENParams`](@ref), [`ContractingRENParams`](@ref), [`LipschitzRENParams`](@ref), [`PassiveRENParams`](@ref).
"""
function DirectParams{T}(
    nu::Int, nx::Int, nv::Int, ny::Int; 
    init = :random,
    polar_param::Bool = true,
    D22_free::Bool = false,
    D22_zero::Bool = false,
    bx_scale::T = T(0), 
    bv_scale::T = T(1), 
    ϵ::T = T(1e-12), 
    rng::AbstractRNG = Random.GLOBAL_RNG
) where T

    # Check options
    if D22_zero
        @warn """Setting D22 fixed at 0. Removing feedthrough."""
        D22_free = true
    end

    # Random sampling
    if init == :random

        B2  = glorot_normal(nx, nu; T=T, rng=rng)
        D12 = glorot_normal(nv, nu; T=T, rng=rng)
        
        ρ = zeros(1)
        
        # Make orthogonal X
        X = glorot_normal(2nx + nv, 2nx + nv; T=T, rng=rng)
        X = Matrix(qr(X).Q)

    # Specify H and compute X
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
        X = Matrix{T}(cholesky(Htild).U) # H = X'*X

    else
        error("Undefined initialisation method ", init)
    end

    # Free parameter for E
    Y1 = glorot_normal(nx, nx; T=T, rng=rng)

    # Output layer
    C2  = glorot_normal(ny,nx; rng=rng)
    D21 = glorot_normal(ny,nv; rng=rng)
    D22 = zeros(T, ny, nu)

    # Parameters for D22 in output layer
    if D22_free
        X3 = zeros(T, 0, 0)
        Y3 = zeros(T, 0, 0)
        Z3 = zeros(T, 0, 0)
    else
        d = min(nu, ny)
        X3 = Matrix{T}(I, d, d)
        Y3 = zeros(T, d, d)
        Z3 = zeros(T, abs(ny - nu), d)
    end
    
    # Bias terms
    bv = T(bv_scale) * glorot_normal(nv; T=T, rng=rng)
    bx = T(bx_scale) * glorot_normal(nx; T=T, rng=rng)
    by = glorot_normal(ny; rng=rng)

    return DirectParams(
        ρ ,X, 
        Y1, X3, Y3, Z3, 
        B2, C2, D12, D21, D22,
        bx, bv, by, T(ϵ), 
        polar_param, D22_free, D22_zero
)
end

function Flux.trainable(L::DirectParams)

    # Different cases for D22 free/zero
    if L.D22_free
        if L.D22_zero
            ps = [L.ρ, L.X, L.Y1, L.B2, L.C2, 
                  L.D12, L.D21, L.bx, L.bv, L.by]
        else
            ps = [L.ρ, L.X, L.Y1, L.B2, L.C2, 
                 L.D12, L.D21, L.D22, L.bx, L.bv, L.by]
        end
    else
        ps = [L.ρ, L.X, L.Y1, L.X3, L.Y3, L.Z3, L.B2,
              L.C2, L.D12, L.D21, L.bx, L.bv, L.by]
    end

    # Don't need ρ if not polar param
    !(L.polar_param) && popfirst!(ps)

    # Removes empty params, useful when nx=0
    return filter(p -> length(p) !=0, ps)
end

function Flux.gpu(M::DirectParams{T}) where T
    # TODO: Test and complete this
    if T != Float32
        println("Moving type: ", T, " to gpu may not be supported. Try Float32!")
    end
    return DirectParams{T}(
        gpu(M.ρ), gpu(M.X), gpu(M.Y1), gpu(M.X3), gpu(M.Y3), 
        gpu(M.Z3), gpu(M.B2), gpu(M.C2), gpu(M.D12), gpu(M.D21),
        gpu(M.D22), gpu(M.bx), gpu(M.bv), gpu(M.by),
        M.ϵ, M.polar_param, M.D22_free, M.D22_zero
    )
end

function Flux.cpu(M::DirectParams{T}) where T
    # TODO: Test and complete this
    return DirectParams{T}(
        cpu(M.ρ), cpu(M.X), cpu(M.Y1), cpu(M.X3), cpu(M.Y3), 
        cpu(M.Z3), cpu(M.B2), cpu(M.C2), cpu(M.D12), cpu(M.D21),
        cpu(M.D22), cpu(M.bx), cpu(M.bv), cpu(M.by),
        M.ϵ, M.polar_param, M.D22_free, M.D22_zero
    )
end

"""
    ==(ps1::DirectParams, ps2::DirectParams)

Define equality for two objects of type `DirectParams`.
    
Checks if all *relevant* parameters are equal. For example, if `D22` is fixed to `0` then the values of `X3, Y3, Z3` are not important and are ignored.
"""
function ==(ps1::DirectParams, ps2::DirectParams)

    # Compare the options
    (ps1.D22_zero != ps2.D22_zero) && (return false)
    (ps1.D22_free != ps2.D22_free) && (return false)
    (ps1.polar_param != ps2.polar_param) && (return false)

    c = fill(false, 15)

    # Check implicit parameters
    c[1] = ps1.X == ps2.X
    c[2] = ps1.Y1 == ps2.Y1

    c[3] = ps1.B2 == ps2.B2
    c[4] = ps1.D12 == ps2.D12

    c[5] = ps1.bx == ps2.bx
    c[6] = ps1.bv == ps2.bv

    c[7] = ps1.ϵ == ps2.ϵ
    c[8] = ps1.polar_param ? (ps1.ρ == ps2.ρ) : true

    if !ps1.D22_free
        c[9] = ps1.X3 == ps2.X3
        c[10] = ps1.Y3 == ps2.Y3
        c[11] = ps1.Z3 == ps2.Z3
        c[12] = true
    else
        c[9], c[10], c[11] = true, true, true
        c[12] = ps1.D22 == ps2.D22
    end

    c[13] = ps1.C2 == ps2.C2
    c[14] = ps1.D21 == ps2.D21
    c[15] = ps1.by == ps2.by

    return all(c)
end
