"""
    glorot_normal(n::Int, m::Int; T=Float64, rng=Random.GLOBAL_RNG)

Generate matrices or vectors from Glorot normal distribution
"""
glorot_normal(n::Int, m::Int; T=Float64, rng=Random.GLOBAL_RNG) = 
    convert.(T, randn(rng, n, m) / sqrt(n + m))
glorot_normal(n::Int; T=Float64, rng=Random.GLOBAL_RNG) = 
    convert.(T, randn(rng, n) / sqrt(n))

"""
    set_output_zero!(m::AbstractRENParams)

Set output map of REN to zero
"""
function set_output_zero!(m::AbstractRENParams)
    m.direct.C2 .*= 0
    m.direct.D21 .*= 0
    m.direct.D22 .*= 0
    m.direct.by .*= 0
end