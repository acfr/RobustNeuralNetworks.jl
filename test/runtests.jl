using RobustNeuralNetworks
using Test

@testset "RobustNeuralNetworks.jl" begin
  
    # Useful
    include("test_utils.jl")

    # Test a basic example from the README
    include("Wrappers/wrap_rens.jl")

    # Test for desired behaviour
    include("ParameterTypes/contraction.jl")
    include("ParameterTypes/general_behavioural_constrains.jl")
    include("ParameterTypes/lipschitz_bound.jl")
    include("ParameterTypes/passivity.jl")

    include("ParameterTypes/dense_lbdn.jl")
end
