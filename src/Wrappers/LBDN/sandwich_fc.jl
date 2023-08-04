# This file is a part of RobustNeuralNetworks.jl. License is MIT: https://github.com/acfr/RobustNeuralNetworks.jl/blob/main/LICENSE 

mutable struct SandwichFC{F, T, D, B}
    σ ::F
    XY::AbstractMatrix{T}       # [X; Y] in the paper
    α ::AbstractVector{T}       # Polar parameterisation
    d ::D                       # d in paper
    b ::B                       # Bias vector
end

"""
    SandwichFC((in, out), σ::F; <keyword arguments>) where F

Construct a non-expansive "sandwich layer" for a dense (fully-connected) LBDN.

A non-expensive layer is a layer with a Lipschitz bound of exactly 1. This layer is provided as a Lipschitz-bounded alternative to [`Flux.Dense`](https://fluxml.ai/Flux.jl/stable/models/layers/#Flux.Dense). Its interface and usage was intentionally designed to be very similar to make it easy to use for anyone familiar with Flux.

# Arguments

- `(in, out)::Pair{<:Integer, <:Integer}`: Input and output sizes of the layer.
- `σ::F=identity`: Activation function.

# Keyword arguments

- `init::Function=Flux.glorot_normal`: Initialisation function for all weights and bias vectors.
- `bias::Bool=true`: Include bias vector or not.
- `output_layer::Bool=false`: Just the output layer of a dense LBDN or regular sandwich layer.
- `T::DataType=Float32`: Data type for weight matrices and bias vectors.
- `rng::AbstractRNG = Random.GLOBAL_RNG`: rng for model initialisation.

# Examples

We can build a dense LBDN directly using `SandwichFC` layers. The model structure is described in Equation 8 of [Wang & Manchester (2023)](https://proceedings.mlr.press/v202/wang23v.html).

```jldoctest
using Flux
using Random
using RobustNeuralNetworks

# Random seed for consistency
rng = Xoshiro(42)

# Model specification
nu = 1                  # Number of inputs
ny = 1                  # Number of outputs
nh = fill(16,2)         # 2 hidden layers, each with 16 neurons
γ = 5                   # Lipschitz bound of 5.0

# Set up dense LBDN model
model = Flux.Chain(
    (x) -> (√γ * x),
    SandwichFC(nu => nh[1], relu; T=Float64, rng),
    SandwichFC(nh[1] => nh[2], relu; T=Float64, rng),
    (x) -> (√γ * x),
    SandwichFC(nh[2] => ny; output_layer=true, T=Float64, rng),
)

# Evaluate on dummy inputs
u = 10*randn(rng, nu, 10)
y = model(u)

println(round.(y;digits=2))

# output

[3.62 4.74 3.58 8.75 3.64 3.0 0.73 1.16 1.0 1.73]
```

See also [`DenseLBDNParams`](@ref), [`DiffLBDN`](@ref).
"""
function SandwichFC(
    (in, out)::Pair{<:Integer, <:Integer},
    σ::F               = identity;
    init::Function     = glorot_normal,
    bias::Bool         = true,
    output_layer::Bool = false, 
    T::DataType        = Float32,
    rng::AbstractRNG   = Random.GLOBAL_RNG,
) where F

    # Matrices and polar param always required
    XY = init(rng, out + in, out)
    α  = [norm(XY)]

    # Only need d/bias if specified (AD knows `nothing` is not differentiable)
    d  = output_layer ? nothing : T.(init(rng, out))
    b  = bias ? T.(init(rng, out)) : nothing

    return SandwichFC{F, T, typeof(d), typeof(b)}(σ, XY, α, d, b)
end

@functor SandwichFC

function (m::SandwichFC)(x::AbstractVecOrMat{T}) where T

    # Extract params
    XY = m.XY
    α = m.α
    d = m.d
    b = m.d
    σ = m.σ
    n = size(XY,2)

    bias = (b !== nothing)
    output = (d === nothing)

    # Get explicit model parameters
    A_T, B_T = norm_cayley(XY, α, n)
    B = B_T'
    
    # Just output layer?
    output && (return bias ? B*x .+ b : B*x)

    # Regular sandwich layer
    Ψd = exp.(d)
    sqrt2 = T(sqrt(2))
    
    x = sqrt2 * (B ./ Ψd) * x
    bias && (x = x .+ b)
    x = sqrt2 * (A_T .* Ψd') * σ.(x)

    return x

end