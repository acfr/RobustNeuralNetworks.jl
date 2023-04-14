module RobustNeuralNetworks

############ Package dependencies ############

using CUDA: CuVector, CuMatrix
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
    abstract type AbstractRENParams{T} end

Direct parameterisation for recurrent equilibrium networks.
"""
abstract type AbstractRENParams{T} end


abstract type AbstractREN end

"""
    abstract type AbstractLBDN end

Parameterisation for Lipschitz-bounded deep networks.
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

export AbstractLBDN
export LBFN

# Functions
export direct_to_explicit
export init_states
export set_output_zero!
export update_explicit!

# Extended functions
# TODO: Need to export things like gpu, cpu, ==, etc.

end # end RobustNeuralNetworks
