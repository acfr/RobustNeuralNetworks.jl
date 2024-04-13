# This file is a part of RobustNeuralNetworks.jl. License is MIT: https://github.com/acfr/RobustNeuralNetworks.jl/blob/main/LICENSE 

"""
    tril_eq_layer(σ::F, D11::Matrix, b::VecOrMat) where F

Evaluate and solve lower-triangular equilibirum layer.
"""
function tril_eq_layer(σ::F, D11, b) where F

    # Solve the equilibirum layer
    w_eq = solve_tril_layer(σ, D11, b)

    # Run the equation for auto-diff to get grads: ∂σ/∂(.) * ∂(D₁₁w + b)/∂(.)
    # By definition, w_eq1 = w_eq so this doesn't change the forward pass.
    v = D11 * w_eq .+ b
    w_eq = σ.(v)
    return tril_layer_back(σ, D11, v, w_eq)
end

"""
    solve_tril_layer(σ::F, D11::Matrix, b::VecOrMat) where F

Solves w = σ.(D₁₁*w .+ b) for lower-triangular D₁₁, where
σ is an activation function with monotone slope restriction (eg: relu, tanh).
"""
function solve_tril_layer(σ::F, D11, b) where F
    z_eq  = similar(b)
    Di_zi = typeof(b)(zeros(Float32, 1, size(b,2))) 
    # similar(b, 1, size(b,2)) can induce NaN on GPU!!!
    for i in axes(b,1)
        Di = @view D11[i:i, 1:i - 1]
        zi = @view z_eq[1:i-1,:]
        bi = @view b[i:i, :]

        mul!(Di_zi, Di, zi)
        z_eq[i:i,:] .= σ.(Di_zi .+ bi)  
    end
    return z_eq
end
@non_differentiable solve_tril_layer(σ, D11, b)

"""
    tril_layer_back(σ::F, D11::Matrix, v::VecOrMat{T}, w_eq::VecOrMat{T}) where {F,T}

Dummy function to force auto-diff engines to use the custom backwards pass.
"""
function tril_layer_back(σ::F, D11, v, w_eq::AbstractVecOrMat{T}) where {F,T}
    return w_eq
end

function rrule(::typeof(tril_layer_back), σ::F, D11, v, w_eq::AbstractVecOrMat{T}) where {F,T}

    # Forwards pass
    y = tril_layer_back(σ, D11, v, w_eq)

    # Reverse mode
    function tril_layer_back_pullback(Δy)

        Δf = NoTangent()
        Δσ = NoTangent()
        ΔD11 = NoTangent()
        Δb = NoTangent()

        # Get gradient of σ(v) wrt v evaluated at v = D₁₁w + b
        _back(σ, v) = rrule(σ, v)[2]
        backs = _back.(σ, v)
        j = map(b -> b(one(T))[2], backs)

        # Compute gradient from implicit function theorem
        Δw_eq = v
        for i in axes(Δw_eq, 2)
            ji = @view j[:, i]
            Δyi = @view Δy[:, i]
            Δw_eq[:,i] = (I - (ji .* D11))' \ Δyi
        end
        return Δf, Δσ, ΔD11, Δb, Δw_eq
    end

    return y, tril_layer_back_pullback
end
