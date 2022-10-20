using RecurrentEquilibriumNetworks
using Test

@testset "RecurrentEquilibriumNetworks.jl" begin
  
    # Test a basic example from the README
    include("wrap_ren.jl")

    # Test for desired behaviour
    include("contraction.jl")
    include("general_behavioural_constrains.jl")
    include("lipschitz_bound.jl")

end
