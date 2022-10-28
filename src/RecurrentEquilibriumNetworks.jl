module RecurrentEquilibriumNetworks

############ Package dependencies ############

using CUDA
using DocStringExtensions
using Flux
using LinearAlgebra
using MatrixEquations: lyapd, plyapd
using Random
using Zygote
using Zygote: @adjoint

import Base.:(==)
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
include("Base/ren.jl")

# Wrappers
include("Wrappers/diff_ren.jl")
include("Wrappers/wrap_ren.jl")
include("Wrappers/wrap_ren_2.jl")

# Variations of REN
include("ParameterTypes/utils.jl")
include("ParameterTypes/contracting_ren.jl")
include("ParameterTypes/general_ren.jl")
include("ParameterTypes/lipschitz_ren.jl")
include("ParameterTypes/passive_ren.jl")

# Main REN type


############ Exports ############

# Types
export AbstractREN
export AbstractRENParams

export DirectParams
export ExplicitParams
export REN

export ContractingRENParams
export GeneralRENParams
export LipschitzRENParams
export PassiveRENParams

export DiffREN
export WrapREN
export WrapREN2

# Functions
export direct_to_explicit
export init_states
export set_output_zero!
export update_explicit!

end # end RecurrentEquilibriumNetworks
