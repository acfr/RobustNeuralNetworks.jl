cd(@__DIR__)
using Pkg
Pkg.activate("../")

using Flux, RobustNeuralNetworks

T  = Float32
nl = Flux.relu
nu, nx, nv, ny = 1, 10, 20, 1
ps = ContractingRENParams{T}(nu, nx, nv, ny; nl)

println(ps.direct) # Access direct params

model = REN(ps)         # Create explicit model
println(model.explicit) # Access explicit params