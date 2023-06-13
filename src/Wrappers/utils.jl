"""
    get_lipschitz(model)

Extract Lipschitz bound from a Lipschitz-bounded model

Returns Lipschitz bound as a float. Function only works on the following types:
- `LBDN` and `DiffLBDN`
- `DenseLBDNParams` and `DirectLBDNParams`
- `LipschitzRENParams`
"""
function get_lipschitz end

get_lipschitz(m::DirectLBDNParams) = exp(m.log_γ[1])
get_lipschitz(m::DenseLBDNParams) = get_lipschitz(m.direct)
get_lipschitz(m::DiffLBDN) = get_lipschitz(m.params)
get_lipschitz(m::LBDN) = m.explicit.sqrtγ^2

get_lipschitz(m::LipschitzRENParams) = m.γ