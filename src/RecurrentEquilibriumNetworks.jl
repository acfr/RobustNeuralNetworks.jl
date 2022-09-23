module RecurrentEquilibriumNetworks

############ Package dependencies ############

using CUDA
using DocStringExtensions
using Flux
using LinearAlgebra
using MatrixEquations: lyapd, plyapd
using Random
using Zygote: @adjoint

import Flux.gpu, Flux.cpu


############ Abstract type ############

"""
$(TYPEDEF)
"""
abstract type AbstractRENParams end


############ Includes ############

# Useful
include("Base/utils.jl")
include("Base/acyclic_ren_solver.jl")

# Common structures
include("Base/direct_params.jl")
include("Base/output_layer.jl")
include("Base/ren.jl")

# Variations of REN
include("ParameterTypes/contracting_ren.jl")
include("ParameterTypes/general_ren.jl")

# Main REN type


############ Exports ############
export AbstractRENParams
export ContractingRENParams
export DirectParams
export ExplicitParams
export GeneralRENParams
export OutputLayer
export REN

end # end RecurrentEquilibriumNetworks
