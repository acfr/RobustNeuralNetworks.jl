# This file is a part of RobustNeuralNetworks.jl. License is MIT: https://github.com/acfr/RobustNeuralNetworks.jl/blob/main/LICENSE 

module RobustNeuralNetworks

############ Package dependencies ############

using ChainRulesCore: NoTangent, @non_differentiable
using Flux: relu, identity, @functor
using LinearAlgebra
using Random
using Zygote: Buffer

import Base.:(==)
import ChainRulesCore: rrule
import Flux: trainable, glorot_normal

# Note: to remove explicit dependency on Flux.jl, use the following
#   using Functors: @functor
#   using NNlib: relu, identity
#   import Optimisers.trainable
# and re-write `glorot_normal` yourself.


############ Abstract types ############

"""
    abstract type AbstractRENParams{T} end

Direct parameterisation for recurrent equilibrium networks.
"""
abstract type AbstractRENParams{T} end

abstract type AbstractREN{T} end

"""
    abstract type AbstractLBDNParams{T, L} end

Direct parameterisation for Lipschitz-bounded deep networks.
"""
abstract type AbstractLBDNParams{T, L} end

abstract type AbstractLBDN{T, L} end


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

include("ParameterTypes/hybrid_passive_ren.jl")
include("ParameterTypes/dense_lbdn.jl")

# Wrappers
include("Wrappers/REN/ren.jl")
include("Wrappers/REN/diff_ren.jl")
include("Wrappers/REN/wrap_ren.jl")

include("Wrappers/LBDN/lbdn.jl")
include("Wrappers/LBDN/diff_lbdn.jl")
include("Wrappers/LBDN/sandwich_fc.jl")

include("Wrappers/utils.jl")


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
export SandwichFC

# Functions
export direct_to_explicit
export get_lipschitz
export init_states
export set_output_zero!
export update_explicit!

end # end RobustNeuralNetworks
