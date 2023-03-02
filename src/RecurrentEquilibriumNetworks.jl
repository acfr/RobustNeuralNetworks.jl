module RecurrentEquilibriumNetworks

############ Package dependencies ############

using CUDA: CuVector, CuMatrix
using DocStringExtensions
using Flux
using LinearAlgebra
using MatrixEquations: lyapd, plyapd
using Random
using Zygote: pullback
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

"""
$(TYPEDEF)
"""
abstract type AbstractLBDN end


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

# LBDN
include("LBDN/lbfn.jl")


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

export AbstractLBDN
export LBFN

# Functions
export direct_to_explicit
export init_states
export set_output_zero!
export update_explicit!

end # end RecurrentEquilibriumNetworks
