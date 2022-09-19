using Random

# Glorot normal functions for vectors and matrices (respectively)
glorot_normal(n; T=Float64, rng=Random.GLOBAL_RNG) = 
    convert.(T, randn(rng, n) / sqrt(n))
glorot_normal(n, m; T=Float64, rng=Random.GLOBAL_RNG) = 
    convert.(T, randn(rng, n, m) / sqrt(n + m))