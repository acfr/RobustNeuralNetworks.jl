module RecurrentEquilibriumNetworks

############ Package dependencies ############

using CUDA
using DocStringExtensions
using Flux
using LinearAlgebra
using MatrixEquations: lyapd, plyapd
using Random

import Flux.gpu, Flux.cpu


############ Abstract type ############

"""
$(TYPEDEF)
"""
abstract type AbstractRENParams end


############ Includes ############

# Useful
include("utils.jl")

# Common structures
include("direct_params.jl")
include("output.jl")

# Variations of REN
include("contracting_ren.jl")

# Main REN type


############ Exports ############
export AbstractRENParams
export ContractingREN
export DirectParams

end # end RecurrentEquilibriumNetworks
