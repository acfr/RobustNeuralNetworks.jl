@doc raw"""
    mutable struct ExplicitRENParams{T}

Explicit REN parameter struct.

These parameters define a recurrent equilibrium network with model inputs and outputs ``u_t, y_t``, neuron inputs and outputs ``v_t,w_t``, and states `x_t`.

```math
\begin{equation*}
\begin{bmatrix}
x_{t+1} \\ v_t \\ y_t
\end{bmatrix}
= 
\begin{bmatrix}
A & B_1 & B_2 \\
C_1 & D_{11} & D_{12} \\
C_2 & D_{21} & D_{22} \\
\end{bmatrix}
\begin{bmatrix}
x_t \\ w_t \\ u_t
\end{bmatrix}
+ 
\begin{bmatrix}
b_x \\ b_v \\ b_y
\end{bmatrix}
\end{equation*}
```

See [Revay et al. (2021)](https://arxiv.org/abs/2104.05942) for more details on explicit parameterisations of REN.
"""
mutable struct ExplicitRENParams{T}
    A  ::AbstractMatrix{T}
    B1 ::AbstractMatrix{T}
    B2 ::AbstractMatrix{T}
    C1 ::AbstractMatrix{T}
    C2 ::AbstractMatrix{T}
    D11::AbstractMatrix{T}
    D12::AbstractMatrix{T}
    D21::AbstractMatrix{T}
    D22::AbstractMatrix{T}
    bx ::AbstractVector{T}
    bv ::AbstractVector{T}
    by ::AbstractVector{T}
end

mutable struct DirectRENParams{T}
    X  ::AbstractMatrix{T}
    Y1 ::AbstractMatrix{T}
    X3 ::AbstractMatrix{T}
    Y3 ::AbstractMatrix{T}
    Z3 ::AbstractMatrix{T}
    B2 ::AbstractMatrix{T}
    C2 ::AbstractMatrix{T}
    D12::AbstractMatrix{T}
    D21::AbstractMatrix{T}
    D22::AbstractMatrix{T}
    bx ::AbstractVector{T}
    bv ::AbstractVector{T}
    by ::AbstractVector{T}
    ϵ  ::T
    ρ  ::AbstractVector{T}              # Used in polar param (if specified)
    polar_param::Bool                   # Whether or not to use polar parameterisation
    D22_free   ::Bool                   # Is D22 free or parameterised by (X3,Y3,Z3)?
    D22_zero   ::Bool                   # Option to remove feedthrough.
    is_output  ::Bool
end

"""
    DirectRENParams{T}(nu, nx, nv; <keyword arguments>) where T

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

- `is_output::Bool=true`: Include output layer ``y_t = C_2 x_t + D_{21} w_t + D_{22} u_t + b_y``.

- `ϵ::T=1e-12`: Regularising parameter for positive-definite matrices.

- `rng::AbstractRNG=Random.GLOBAL_RNG`: rng for model initialisation.

See [Revay et al. (2021)](https://arxiv.org/abs/2104.05942) for parameterisation details.

See also [`GeneralRENParams`](@ref), [`ContractingRENParams`](@ref), [`LipschitzRENParams`](@ref), [`PassiveRENParams`](@ref).
"""
function DirectRENParams{T}(
    nu::Int, nx::Int, nv::Int, ny::Int; 
    init                = :random,
    polar_param::Bool   = true,
    D22_free::Bool      = false,
    D22_zero::Bool      = false,
    bx_scale::T         = T(0), 
    bv_scale::T         = T(1), 
    is_output::Bool     = true,
    ϵ::T                = T(1e-12), 
    rng::AbstractRNG    = Random.GLOBAL_RNG
) where T

    # Check options
    if D22_zero
        @warn """Setting D22 fixed at 0. Removing feedthrough."""
        D22_free = true
    end
    if !is_output
        D22_zero = true
        D22_free = true
        if nx != ny
            @warn """Requested no output layer (identity). Setting `ny = nx`, setting D22 fixed at 0."""
            ny = nx
        end
    end

    # Random sampling
    if init == :random

        B2  = glorot_normal(nx, nu; T=T, rng=rng)
        D12 = glorot_normal(nv, nu; T=T, rng=rng)
        
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
        
        X = Matrix{T}(cholesky(Htild).U) # H = X'*X

    else
        error("Undefined initialisation method ", init)
    end

    # Polar parameter
    ρ = [norm(X)]

    # Free parameter for E
    Y1 = glorot_normal(nx, nx; T=T, rng=rng)

    # Output layer
    if is_output
        C2  = glorot_normal(ny,nx; rng=rng)
        D21 = glorot_normal(ny,nv; rng=rng)
        D22 = zeros(T, ny, nu)
    else
        C2  = Matrix{T}(I, nx, nx)
        D21 = zeros(T, ny, nv)
        D22 = zeros(T, ny, nu)
    end

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
    by = is_output ? glorot_normal(ny; rng=rng) : zeros(T, ny)

    return DirectRENParams(
        X, 
        Y1, X3, Y3, Z3, 
        B2, C2, D12, D21, D22,
        bx, bv, by, T(ϵ), ρ,
        polar_param, D22_free, D22_zero,
        is_output
    )
end

Flux.@functor DirectRENParams

function Flux.trainable(m::DirectRENParams)

    # Field names of trainable params, exclude ρ if needed
    if !m.is_output
        fs = [:X, :Y1, :B2, :D12, :bx, :bv, :ρ]
    elseif m.D22_free
        if m.D22_zero
            fs = [:X, :Y1, :B2, :C2, :D12, :D21, :bx, :bv, :by, :ρ]
        else
            fs = [:X, :Y1, :B2, :C2, :D12, :D21, :D22, :bx, :bv, :by, :ρ]
        end
    else
        fs = [:X, :Y1, :X3, :Y3, :Z3, :B2, :C2, :D12, :D21, :bx, :bv, :by, :ρ]
    end
    !(m.polar_param) && pop!(fs)

    # Get params, ignore empty ones (eg: when nx=0)
    ps = [getproperty(m, f) for f in fs]
    indx = length.(ps) .!= 0
    ps, fs = ps[indx], fs[indx]

    # Flux.trainable() must return a NamedTuple
    return NamedTuple{tuple(fs...)}(ps)
end

function Flux.gpu(M::DirectRENParams{T}) where T
    # TODO: Test and complete this
    if T != Float32
        println("Moving type: ", T, " to gpu may not be supported. Try Float32!")
    end
    return DirectRENParams{T}(
        gpu(M.ρ), gpu(M.X), gpu(M.Y1), gpu(M.X3), gpu(M.Y3), 
        gpu(M.Z3), gpu(M.B2), gpu(M.C2), gpu(M.D12), gpu(M.D21),
        gpu(M.D22), gpu(M.bx), gpu(M.bv), gpu(M.by),
        M.ϵ, M.polar_param, M.D22_free, M.D22_zero
    )
end

function Flux.cpu(M::DirectRENParams{T}) where T
    # TODO: Test and complete this
    return DirectRENParams{T}(
        cpu(M.ρ), cpu(M.X), cpu(M.Y1), cpu(M.X3), cpu(M.Y3), 
        cpu(M.Z3), cpu(M.B2), cpu(M.C2), cpu(M.D12), cpu(M.D21),
        cpu(M.D22), cpu(M.bx), cpu(M.bv), cpu(M.by),
        M.ϵ, M.polar_param, M.D22_free, M.D22_zero
    )
end

"""
    ==(ps1::DirectRENParams, ps2::DirectRENParams)

Define equality for two objects of type `DirectRENParams`.
    
Checks if all *relevant* parameters are equal. For example, if `D22` is fixed to `0` then the values of `X3, Y3, Z3` are not important and are ignored.
"""
function ==(ps1::DirectRENParams, ps2::DirectRENParams)

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
