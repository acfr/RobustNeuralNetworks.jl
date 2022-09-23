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
include("Base/utils.jl")

# Common structures
include("Base/direct_params.jl")
include("Base/output_layer.jl")

# Variations of REN
include("ParameterTypes/contracting_ren.jl")
include("ParameterTypes/general_ren.jl")

# Main REN type


############ Exports ############
export AbstractRENParams
export ContractingRENParams
export DirectParams
export GeneralRENParams
export OutputLayer

end # end RecurrentEquilibriumNetworks
