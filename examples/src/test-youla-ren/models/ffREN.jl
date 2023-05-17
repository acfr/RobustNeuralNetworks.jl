using LinearAlgebra
using Flux
import Flux.gpu, Flux.cpu

# includet("./solvers.jl")
includet("./fixed_point_layer_general.jl")
includet("./output_layer.jl")

unzip(a) = (getfield.(a, x) for x in fieldnames(eltype(a)))


# Implicit, direct parametrizations
abstract type implicit_param end

# Explicit types that we simulate
mutable struct explicit_ff_cell
    ϕ
    A
    B1
    B2
    C1
    D11
    D12
    bx
    bv
end

Flux.trainable(L::explicit_ff_cell) = [L.A, L.B1, L.B2, L.C1, L.D11, L.D12, L.bx, L.bv]

## Stable REN cell
mutable struct implicit_ff_cell{T} <: implicit_param
    ϕ
    V::AbstractMatrix{T}
    S_1::AbstractMatrix{T}
    B2::AbstractMatrix{T}
    D12::AbstractMatrix{T}
    bx::AbstractVector{T}
    bv::AbstractVector{T}
    ϵ::T
end
Flux.trainable(L::implicit_ff_cell) = [L.V, L.S_1, L.B2, L.D12, L.bx, L.bv]

function Flux.gpu(M::implicit_ff_cell{T}) where T
    if T != Float32
        println("Moving type: ", T, " to gpu may not be supported. Try Float32!")
    end
    return implicit_ff_cell{T}(M.ϕ, gpu(M.V), gpu(M.S_1), gpu(M.B2),
                             gpu(M.D12), gpu(M.bx), gpu(M.bv), M.ϵ)
end

function Flux.cpu(M::implicit_ff_cell{T}) where T
    return implicit_ff_cell{T}(M.ϕ, cpu(M.V), cpu(M.S_1), cpu(M.B2),
                             cpu(M.D12), cpu(M.bx), cpu(M.bv), M.ϵ)
end

function implicit_ff_cell{T}(nu, nx, nv; nl=relu, ϵ=0.01, bx_scale=0.0, bv_scale=1.0) where T
    glorot_normal(n, m) = convert.(T, randn(n, m) / sqrt(n + m))
    E = Matrix{T}(I, nx, nx)
    F = Matrix{T}(I, nx, nx)
    P = Matrix{T}(I, nx, nx)
    B1 = zeros(T, nx, nv)
    B2 = glorot_normal(nx, nu)
 
    C1 = zeros(T, nv, nx)
    D11 = glorot_normal(nv, nv)
    D12 = zeros(T, nv, nu)

    Λ = I  #Choose sum abs values to make diagonally dominant
    H22 = 2Λ - D11 - D11'
    Htild = [(E + E' - P) -C1' F';
             -C1 H22 B1'
             F B1  P] + ϵ * I

    S_1 = glorot_normal(nx, nx)

    V = Matrix{T}(cholesky(Htild).U) # H = V'*V
    
    bv = convert.(T, bv_scale * randn(nv) / sqrt(nv))
    bx = convert.(T, bx_scale * rand(nx) / sqrt(nx))

    return implicit_ff_cell{T}(nl, V, S_1, B2, D12, bx, bv, ϵ)
end
# constructor if type is not specified
implicit_ff_cell(nu, nx, nv; nl=relu, ϵ=0.01, bx_scale=0.0, bv_scale=1.0) = implicit_ff_cell{Float64}(nu, nx, nv; nl=nl, ϵ=ϵ, bx_scale=bx_scale, bv_scale=bv_scale)


function sample_ff_ren(nu, nx, nv; nl=relu, ϵ=0.01, bx_scale=0.0, bv_scale=1.0)
    # glorot_normal(n, m) = randn(n, m) / sqrt(n + m)
    B2 = randn(nx, nu) / sqrt(nx+nu)
 
    D12 = randn(nv, nu) / sqrt(nv+nu)
    V = 2 * randn(2nx + nv, 2nx + nv) / sqrt(2nx + nv)
    
    bv = bv_scale * randn(nv) / sqrt(nv)
    bx = bx_scale * rand(nx) / sqrt(nx)

    S_1 = randn(nx, nx) / sqrt(nx+nx)

    return implicit_ff_cell(nl, V, S_1, B2, D12, bx, bv, ϵ)
end

function explicit(model::implicit_ff_cell)
    nx = size(model.B2, 1)
    nu = size(model.B2, 2)
    nv = size(model.D12, 1)

    H = model.V' * model.V + model.ϵ * I

    # For some reason taking view doesn't work with CUDA
    H11 = H[1:nx, 1:nx]
    H22 = H[ nx + 1:nx + nv, nx + 1:nx + nv]
    H33 = H[ nx + nv + 1:2nx + nv, nx + nv + 1:2nx + nv]
    H21 = H[nx + 1:nx + nv, 1:nx]
    H31 = H[ nx + nv + 1:2nx + nv, 1:nx]
    H32 = H[ nx + nv + 1:2nx + nv, nx + 1:nx + nv]
    
    # Implicit model parameters
    S_1 = (model.S_1 - model.S_1') / 2    
    
    B2 = model.B2
    D12 = model.D12
   
    P = H33
    E = (H11 + P + S_1) / 2
    F = H31

    # equilibrium network stuff
    C1 = - (H21)
    B1 = H32
    Λᵢ = diag(H22) / 2
    D11 = -tril(H22, -1)  # strictly lower triangular part

    # Construct explicit rnn model. 
    # Use \bb to differentiate from implicit model   
    𝔸 = E \ F
    𝔹_1 = E \ B1
    𝔹_2 = E \ B2

    ℂ_1 = (1 ./ Λᵢ) .* C1
    𝔻_11 = (1 ./ Λᵢ) .* D11
    𝔻_12 = (1 ./ Λᵢ) .* D12
    
    bx = model.bx
    bv = model.bv

    return explicit_ff_cell(model.ϕ, 𝔸, 𝔹_1, 𝔹_2, ℂ_1, 𝔻_11, 𝔻_12, bx, bv)
end

function init_state(model::implicit_ff_cell{T}, batches) where T
    nx = size(model.B2, 1)
    return typeof(model.V)(zeros(T, nx, batches))
end

function (exp_cell::explicit_ff_cell)(xt, ut)
    b = exp_cell.C1 * xt + exp_cell.D12 * ut .+ exp_cell.bv
    wt = tril_eq_layer(exp_cell.ϕ, exp_cell.D11, b)
    xn = exp_cell.A * xt + exp_cell.B1 * wt + exp_cell.B2 * ut .+ exp_cell.bx
    return xn, (xt, wt)
end

function simulate(exp_cell::explicit_ff_cell, x0, ut)
    recurrent = Flux.Recur(exp_cell, x0)
    return unzip(recurrent.(ut))
end


function (implicit_cell::implicit_ff_cell)(xt, ut)
    exp_cell = explicit(implicit_cell)
    return exp_cell(xt, ut)        
end

function simulate(cell::implicit_ff_cell, x0, ut)
    exp_cell = explicit(cell)
    return simulate(exp_cell, x0, ut)
end

# Dissipative feedforward RNN code

mutable struct dissipative_ff_rnn{T}
    nu
    nx
    nv
    ny
    implicit_cell::implicit_ff_cell{T}
    output::output{T}
    Q::Matrix{T}
    S::Matrix{T}
    R::Matrix{T}
end

Flux.trainable(model::dissipative_ff_rnn) = (Flux.trainable(model.implicit_cell)..., Flux.trainable(model.output)...)

# Constructors
function dissipative_ff_rnn{T}(nu, nx, nv, ny, Q, S, R; nl=relu, ϵ=0.001, bx_scale=0.0, bv_scale=1.0) where T
    cell_params = implicit_ff_cell{T}(nu, nx, nv; nl=nl, ϵ, bx_scale, bv_scale)
    output_params = output{T}(nu, nx, nv, ny)
    return dissipative_ff_rnn{T}(nu, nx, nv, ny, cell_params, output_params, Q, S, R)
end

function init_state(model::dissipative_ff_rnn{T}, batches) where T
    return typeof(model.implicit_cell.V)(zeros(T, model.nx, batches))
end

# Common forms of dissipativity
function bounded_ff_rnn(T::DataType, nu, nx, nv, ny, γ; 
                        nl=relu, ϵ=0.001, bx_scale=0.0, bv_scale=1.0)
    R = Matrix{T}(γ * I, nu, nu)
    S = zeros(T, nu, ny)
    Q = Matrix{T}(-I / γ, ny, ny)
    return dissipative_ff_rnn{T}(nu, nx, nv, ny, Q, S, R; nl=relu, ϵ=0.001, bx_scale=0.0, bv_scale=1.0) 
end

function stable_ff_rnn(T::DataType, nu, nx, nv, ny; 
                        nl=relu, ϵ=0.001, bx_scale=0.0, bv_scale=1.0)
    return bounded_ff_rnn(T, nu, nx, nv, ny, Inf; nl=nl, ϵ=ϵ, bx_scale=bx_scale, bv_scale=bv_scale) 
end

function passive_ff_rnn(T::DataType, nu, nx, nv, ny; 
                     η=0.0, nl=relu, ϵ=0.001, bx_scale=0.0, bv_scale=1.0)

    R = zeros(T, nu, nu)
    S = Matrix(I / 2, nu, ny)
    Q = Matrix(-η * I, nu, nu)
    return dissipative_ff_rnn(nu, nx, nv, ny, Q, S, R; nl=relu, ϵ=0.001, bx_scale=0.0, bv_scale=1.0) 
end

# default type is Float64
bounded_ff_rnn(nu, nx, nv, ny, γ; nl=relu, ϵ=0.001, bx_scale=0.0, bv_scale=1.0) = 
                bounded_ff_rnn(Float64, nu, nx, nv, ny, γ; 
                            nl=nl, ϵ=ϵ, bx_scale=bx_scale, bv_scale=bv_scale)


stable_ff_rnn(nu, nx, nv, ny; nl=relu, ϵ=0.001, bx_scale=0.0, bv_scale=1.0) = 
                stable_ff_rnn(Float64, nu, nx, nv, ny; 
                                nl=nl, ϵ=ϵ, bx_scale=bx_scale, bv_scale=bv_scale)

passive_ff_rnn(nu, nx, nv, ny; η=0.0, nl=relu, ϵ=0.001, bx_scale=0.0, bv_scale=1.0) = 
                passive_ff_rnn(Float64, nu, nx, nv, ny; 
                                        η=η, nl=nl, ϵ=ϵ, bx_scale=bx_scale, bv_scale=bv_scale)
            
                                       
# Construct and explicit 
function explicit(model::dissipative_ff_rnn)
    nx = model.nx
    nu = model.nu
    ny = model.ny
    nv = model.nv

    # dissipation parameter
    Q = model.Q
    S = model.S
    R = model.R
    
    # Implicit model parameters
    S_1 = (model.implicit_cell.S_1 - model.implicit_cell.S_1') / 2
    
    C2 = model.output.C2
    D21 = model.output.D21
    D22 = model.output.D22

    B2 = model.implicit_cell.B2
    D12 = model.implicit_cell.D12
    
    # RHS of dissipation inequality
    Γ1 = [C2'; D21'; zeros(nx, ny)] * Q * [C2 D21 zeros(ny, nx)]  # possibly transpose zeros
    Γ2 = [(C2' * S'); (D21' * S' - D12); B2] * inv(R) * [(S * C2) (S * D21 - D12') B2']

    H = model.implicit_cell.V' * model.implicit_cell.V - Γ1 + Γ2 + model.implicit_cell.ϵ * I 
    H11 = H[1:nx, 1:nx]
    H22 = H[ nx + 1:nx + nv, nx + 1:nx + nv]
    H33 = H[ nx + nv + 1:2nx + nv, nx + nv + 1:2nx + nv]
    H21 = H[nx + 1:nx + nv, 1:nx]
    H31 = H[ nx + nv + 1:2nx + nv, 1:nx]
    H32 = H[ nx + nv + 1:2nx + nv, nx + 1:nx + nv]

    # extract parameters from implicit rnn
    Λᵢ = diag(H22) / 2
    D11 = -tril(H22, -1)
    P = H33
    E = (H11 + P + S_1) / 2
    F = H31
    C1 = -H21
    B1 = H32
    
    # Construct explicit rnn model. Use \bb font to differentiate
    A = E \ F
    𝔹_1 = E \ B1
    𝔹_2 = E \ B2

    ℂ_1 = (1 ./ Λᵢ) .* C1
    𝔻_11 = (1 ./ Λᵢ) .* D11
    𝔻_12 = (1 ./ Λᵢ) .* D12
    
    bx = model.implicit_cell.bx
    bv = model.implicit_cell.bv
    
    return explicit_ff_cell(model.implicit_cell.ϕ, A, 𝔹_1, 𝔹_2, ℂ_1, 𝔻_11, 𝔻_12, bx, bv)
end

function (model::dissipative_ff_rnn)(x0, ut)
    exp_cell = explicit(model)
    xt, wt = simulate(exp_cell, x0, ut)
    return model.output.(xt, wt, ut), xt[end]
end


function check_lmi(model::dissipative_ff_rnn)
    nx = model.nx
    nu = model.nu
    ny = model.ny
    nv = model.nv

    # dissipation parameter
    Q = model.Q
    S = model.S
    R = model.R
    
    # Implicit model parameters
    S_1 = (model.implicit_cell.S_1 - model.implicit_cell.S_1') / 2
    
    C2 = model.output.C2
    D21 = model.output.D21
    D22 = model.output.D22

    B2 = model.implicit_cell.B2
    D12 = model.implicit_cell.D12
    
    # RHS of dissipation inequality
    Γ1 = [C2'; D21'; zeros(nx, ny)] * Q * [C2 D21 zeros(ny, nx)]  # possibly transpose zeros
    Γ2 = [(C2' * S'); (D21' * S' - D12); B2] * inv(R) * [(S * C2) (S * D21 - D12') B2']

    H = model.implicit_cell.V' * model.implicit_cell.V - Γ1 + Γ2 + model.implicit_cell.ϵ * I 
    H11 = H[1:nx, 1:nx]
    H22 = H[ nx + 1:nx + nv, nx + 1:nx + nv]
    H33 = H[ nx + nv + 1:2nx + nv, nx + nv + 1:2nx + nv]
    H21 = H[nx + 1:nx + nv, 1:nx]
    H31 = H[ nx + nv + 1:2nx + nv, 1:nx]
    H32 = H[ nx + nv + 1:2nx + nv, nx + 1:nx + nv]

    # extract parameters from implicit rnn
    Λᵢ = diag(H22) / 2
    Λ = Diagonal(Λᵢ)
    D11 = -tril(H22, -1)
    P = H33
    E = (H11 + P + S_1) / 2
    F = H31
    C1 = -H21
    B1 = H32

    # Construct the LMI
    lmi = [(E + E' - P + C2' * Q * C2) (C2' * Q * D21 - C1') (C2' * S') F';
           (D21' * Q * C2 - C1) (2Λ - D11 - D11' + D21' * Q * D21) (D21' * S' - D12) B1';
           (S * C2) (S * D21 - D12') R B2';
           F B1 B2 P]
end


function test_ff_REN()

    device = cpu
    # Check equilibrium solve accuracy
    nu = 5
    model = implicit_ff_cell{Float64}(5, 5, 10; nl=tanh) |> device
    nPoints = 100

    x0 = init_state(model, nPoints) |> device
    utrain = randn(nu, nPoints) |> device
    ytrain = sin.(utrain)

    exp_cell = explicit(model)

    xn, (xt, wt) = exp_cell(x0, utrain)

    b = exp_cell.C1 * x0 + exp_cell.D12 * utrain .+ exp_cell.bv
    err = norm(wt - exp_cell.ϕ.(exp_cell.D11 * wt + b))

    if err > 1E-8
        println("Forward solve doesnt seem to be working well")
    end
    
    # check gradient calculation via FD
    sample_loss(y1, y2) = norm(y1 - y2)^2
    L() = mean(sample_loss.(ytrain, model(x0, utrain)[1]))
    fd_test_grads(L, Flux.params(model))  # param gradients
    fd_test_grads(L, Flux.Params([x0, utrain]))  # input gradients

    
    # Try simple noiseless regression task
    model = implicit_ff_cell{Float32}(1, 1, 40; nl=tanh, ϵ=1.0) |> device

    nPoints = 50
    utrain = 2 * randn(1, nPoints) |> device
    ytrain = (utrain.^3)
    x0 = init_state(model, nPoints)
    

    loss() = norm(ytrain - model(x0, utrain)[1])
    ps = Flux.Params(Flux.trainable(model))
    
    opt = Flux.Optimise.ADAM(1E-3)
    for k in 1:2000
        train_loss, back = Zygote.pullback(loss, ps)

        # calculate gradients and update loss
        ∇J = back(one(train_loss))
        update!(opt, ps, ∇J)
        
        printfmt("Iteration: {1:2d}\tTraining loss: {2:1.2E}\n", k, train_loss)
    end

    exp_cell = explicit(model)
    ϕ = exp_cell.ϕ
    W = exp_cell.D11
    b = exp_cell.C1 * x0 + exp_cell.D12 * utrain .+ exp_cell.bv

    xn, (xt, wt) = model(x0, utrain)
    err = norm(wt - ϕ.(W * wt + b))
    zeq = wt

    # Run to test if gradient is wrong
    fd_test_grads(loss, Flux.params(model))  # param gradients
    fd_test_grads(loss, Flux.Params([x0, utrain]))  # input gradients
    

    yest, (xt, wt) = model(x0, utrain)

    plot(utrain', ytrain'; seriestype=:scatter)
    plot!(utrain', yest'; seriestype=:scatter)

end