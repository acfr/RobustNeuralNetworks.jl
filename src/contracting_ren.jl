include("./direct_params.jl")
include("./output.jl")
include("./utils.jl")

mutable struct ContractingREN{T} <: DirectREN
    nu::Int
    nx::Int
    nv::Int
    ny::Int
    direct_params::DirectParams{T}
    output::Output{T}
end