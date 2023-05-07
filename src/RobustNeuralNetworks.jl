module RobustNeuralNetworks

############ Package dependencies ############

using CUDA: CuVector, CuMatrix
using Flux
using LinearAlgebra
using MatrixEquations: lyapd, plyapd
using Random
using Zygote: pullback, Buffer
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
    abstract type AbstractLBDNParams{T} end

Direct parameterisation for Lipschitz-bounded deep networks.
"""
abstract type AbstractLBDNParams{T} end

"""
    abstract type AbstractLBDN{T} end

Parameterisation for Lipschitz-bounded deep networks.
"""
abstract type AbstractLBDN{T} end


############ Includes ############

# Useful
include("Base/utils.jl")
include("Base/acyclic_ren_solver.jl")

# Common structures
include("Base/ren_params.jl")
include("Base/lbdn_params.jl")

# Variations of REN
include("ParameterTypes/utils.jl")
include("ParameterTypes/contracting_ren.jl")
include("ParameterTypes/general_ren.jl")
include("ParameterTypes/lipschitz_ren.jl")
include("ParameterTypes/passive_ren.jl")

include("ParameterTypes/dense_lbdn.jl")

# Wrappers
include("Wrappers/REN/ren.jl")
include("Wrappers/REN/diff_ren.jl")
include("Wrappers/REN/wrap_ren.jl")

include("Wrappers/LBDN/lbdn.jl")
include("Wrappers/LBDN/diff_lbdn.jl")


############ Exports ############

# Abstract types
export AbstractRENParams
export AbstractREN

export AbstractLBDNParams
export AbstractLBDN

# Basic types
export DirectRENParams
export ExplicitRENParams

export DirectLBDNParams
export ExplicitLBDNParams

# Parameter types
export ContractingRENParams
export GeneralRENParams
export LipschitzRENParams
export PassiveRENParams

export DenseLBDNParams

# Wrappers
export REN
export DiffREN
export WrapREN

export LBDN
export DiffLBDN

# Functions
export direct_to_explicit
export init_states
export set_output_zero!
export update_explicit!

# Extended functions
# TODO: Need to export things like gpu, cpu, ==, etc.

end # end RobustNeuralNetworks
