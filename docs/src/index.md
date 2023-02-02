# RecurrentEquilibriumNetworks.jl Documentation

*Bringing robust machine learning tools to Julia.*

```@contents
```

## Types

```@docs
DirectParams
```

## Functions

```@docs
hmatrix_to_explicit
```

## Examples

Here's an example. You should not be able to see the import statement of `Random`.
```@example test
using Random
a = 1
b = 2*rand()
2a + b
```
Can we continue using variables from this example?
```@example test
println(a+b)
```
We can even make things look like the REPL.
```@repl
a = 1
b = 2
a + b
```

We can even delay execution of an example over a few different example blocks. Start a for loop here...
```@example half-loop; continued = true
for i in 1:3
    j = i^2
```
Then write something insightful and finish it below...
```@example half-loop
    println(j)
end
```

It's worth having a look at the `@setup` macro as well when you can. It will make it much easier to write examples that include a number of lines of setup which should be hidden. Having said that, it might be useful to show the reader how you set up the example!

Most of your examples should be written with the `@jldoctest` macro. I'll give it a go below, but have a look at how `ControlSystems.jl` does things too.

Example:
```@jldoctest TESTING
using Random
using RecurrentEquilibriumNetworks

batches = 50
nu, nx, nv, ny = 4, 10, 20, 2

contracting_ren_ps = ContractingRENParams{Float64}(nu, nx, nv, ny)
contracting_ren = REN(contracting_ren_ps)

x0 = init_states(contracting_ren, batches)
u0 = randn(contracting_ren.nu, batches)

x1, y1 = contracting_ren(x0, u0)  # Evaluates the REN over one timestep

println(size(y1))

# output

(2, 50)
```


## Index

```@index
```

## Docstrings

Work on the presentation of this a bit....
```@autodocs
Modules = [RecurrentEquilibriumNetworks]
Private = false
```

## TODO:
- Add a logo
- Fill out this main documentation page
- See [ControlSystems.jl](https://juliacontrol.github.io/ControlSystems.jl/stable/) for a good example of how to structure this page.