# This file is a part of RobustNeuralNetworks.jl. License is MIT: https://github.com/acfr/RobustNeuralNetworks.jl/blob/main/LICENSE 

using RobustNeuralNetworks
using Test

@testset "RobustNeuralNetworks.jl" begin
  
    # Useful
    include("test_utils.jl")

    # Test for desired behaviour
    include("ParameterTypes/contraction.jl")
    include("ParameterTypes/general_behavioural_constrains.jl")
    include("ParameterTypes/lipschitz_bound.jl")
    include("ParameterTypes/passivity.jl")

    include("ParameterTypes/dense_lbdn.jl")

    # Test wrappers
    include("Wrappers/wrap_ren.jl")
    include("Wrappers/diff_ren.jl")
    include("Wrappers/diff_lbdn.jl")
    include("Wrappers/zero_dim.jl")

end
