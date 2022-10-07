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


############ Abstract types ############

"""
$(TYPEDEF)
"""
abstract type AbstractRENParams{T} end

"""
$(TYPEDEF)
"""
abstract type AbstractREN end


############ Includes ############

# Useful
include("Base/utils.jl")
include("Base/acyclic_ren_solver.jl")

# Common structures
include("Base/direct_params.jl")
include("Base/output_layer.jl")
include("Base/ren.jl")
include("Base/wrapren.jl")

# Variations of REN
include("ParameterTypes/utils.jl")
include("ParameterTypes/contracting_ren.jl")
include("ParameterTypes/general_ren.jl")

# Main REN type


############ Exports ############

# Types
export AbstractREN
export AbstractRENParams
export ContractingRENParams
export DirectParams
export ExplicitParams
export GeneralRENParams
export OutputLayer
export REN
export WrapREN

# Functions
export init_states
export set_output_zero!
export update_explicit!

end # end RecurrentEquilibriumNetworks
