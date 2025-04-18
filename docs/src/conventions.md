# [Conventions](@id conventions)

## Spatial dimensions and directions

We use the following numbering schemes on Cartesian or curved structured meshes.
- The `orientation`s are numbered as
  `1 => x, 2 => y, 3 => z`.
  For example, numerical fluxes such as
  `flux_central(u_ll, u_rr, orientation, equations::AbstractEquations)`
  use the `orientation` in this way.
- The `direction`s are numbered as
  `1 => -x, 2 => +x, 3 => -y, 4 => +y, 5 => -z, 6 => +z`.
  For example, the `boundary_conditions` are ordered in this way
  when a `Tuple` of boundary conditions per direction is passed
  to the constructor of a `SemidiscretizationHyperbolic`.
- For structured and unstructured curved meshes the concept of direction is
  generalized via the variable `normal_direction`. This variable points in the
  normal direction at a given, curved surface. For the computation of boundary fluxes
  the `normal_direction` is normalized to be a `normal_vector` used, for example, in
  [`FluxRotated`](@ref).


## Cells vs. elements vs. nodes

To uniquely distinguish between different components of the discretization, we use the
following naming conventions:
* The computational domain is discretized by a `mesh`, which is made up of
  individual `cell`s. In general, neither the `mesh` nor the `cell`s should be
  aware of any solver-specific knowledge, i.e., they should not store anything
  that goes beyond the geometrical information and the connectivity.
* The numerical `solver`s do not directly store their information inside the `mesh`,
  but use own data structures. Specifically, for each `cell` on which
  a solver wants to operate, the solver creates an `element` to store
  solver-specific data.
* For discretization schemes such as the discontinuous Galerkin or the finite
  element method, inside each `element` multiple `nodes` may be defined, which
  hold nodal information. The nodes are again a solver-specific component, just
  like the elements.
* We often identify `element`s, `node`s, etc. with their (local or global)
  integer index. Convenience iterators such as `eachelement`, `eachnode`
  use these indices.


## Keywords in elixirs

Trixi.jl is distributed with several examples in the form of elixirs, small
Julia scripts containing everything to set up and run a simulation. Working
interactively from the Julia REPL with these scripts can be quite convenient
while for exploratory research and development of Trixi.jl. For example, you
can use the convenience function
[`trixi_include`](@ref)
to `include` an elixir with some modified arguments. To enable this, it is
helpful to use a consistent naming scheme in elixirs, since
[`trixi_include`](@ref)
can only perform simple replacements. Some standard variables names are

- `polydeg` for the polynomial degree of a solver
- `surface_flux` for the numerical flux at surfaces
- `volume_flux` for the numerical flux used in flux differencing volume terms

Moreover, [`convergence_test`](@ref) requires that the spatial resolution is
set via the keywords
- `initial_refinement_level`
  (an integer, e.g., for the [`TreeMesh`](@ref) and the [`P4estMesh`](@ref)) or
- `cells_per_dimension`
  (a tuple of integers, one per spatial dimension, e.g., for the [`StructuredMesh`](@ref)
  and the [`DGMultiMesh`](@ref)).


## Variable names

- Use descriptive names (using `snake_case` for variables/functions and `CamelCase` for types)
- Use a suffix `_` as in `name_` for local variables that would otherwise hide existing symbols.
- Use a prefix `_` as in `_name` to indicate internal methods/data that are "fragile" in the
  sense that there's no guarantee that they might get changed without notice. These are also not
  documented with a docstring (but maybe with comments using `#`).


## Array types and wrapping

To allow adaptive mesh refinement efficiently when using time integrators from
[OrdinaryDiffEq](https://github.com/SciML/OrdinaryDiffEq.jl) and its sub-packages,
Trixi.jl allows to represent numerical solutions in two different ways. Some discussion
can be found [online](https://github.com/SciML/OrdinaryDiffEq.jl/pull/1275) and
in form of comments describing `Trixi.wrap_array` and `Trixi.wrap_array_native`
in the source code of Trixi.jl.
The flexibility introduced by this possible wrapping enables additional
[performance optimizations](https://github.com/trixi-framework/Trixi.jl/pull/509).
However, it comes at the cost of some additional abstractions (and needs to be
used with caution, as described in the source code of Trixi.jl). Thus, we use the
following conventions to distinguish between arrays visible to the time integrator
and wrapped arrays mainly used internally.

- Arrays visible to the time integrator have a suffix `_ode`, e.g., `du_ode`, `u_ode`.
- Wrapped arrays do not have a suffix, e.g., `du, u`.

Methods either accept arrays visible to the time integrator or wrapped arrays
based on the following rules.
- When some solution is passed together with a semidiscretization `semi`, the
  solution must be a `u_ode` that needs to be  wrapped via `wrap_array(u_ode, semi)`
  (or `wrap_array_native(u_ode, semi)`) for further processing.
- When some solution is passed together with the `mesh, equations, solver, cache, ...`,
  it is already wrapped via `wrap_array` (or `wrap_array_native`).
- Exceptions of this rule are possible, e.g., for AMR, but must be documented in
  the code.
- `wrap_array` should be used as default option. `wrap_array_native` should only
  be used when necessary, e.g., to avoid additional overhead when interfacing
  with external C libraries such as HDF5, MPI, or visualization.

## Numeric types and type stability

In Trixi.jl, we use generic programming to support custom data types to store the numerical simulation data, including standard floating point types and automatic differentiation types.
Specifically, `Float32` and `Float64` types are fully supported, including the ability to run Trixi.jl on hardware that only supports `Float32` types.
We ensure the type stability of these numeric types throughout the development process.
Below are some guidelines to apply in various scenarios.

### Exact floating-point numbers

Some real numbers can be represented exactly as both `Float64` and `Float32` values (e.g., `0.25`, `0.5`, `1/2`). We prefer to use `Float32` type for these numbers to achieve a concise way of possible type promotion. For example,
```julia
# Assume we have `0.25`, `0.5`, `1/2` in function
0.25f0, 0.5f0, 0.5f0 # corresponding numbers
```
Generally, this equivalence is true for integer multiples of powers of two. That is, numbers that can be written as ``m 2^n``, where ``m, n \in \mathbb{Z}``, and where ``m`` and ``n`` are such that the result is representable as a [single precision floating point](https://en.wikipedia.org/wiki/Single-precision_floating-point_format) value. If a decimal value `v` is exactly representable in `Float32`, the expression
```julia
Float32(v) == v
```
will evaluate to `true`.

### Non-exact floating-point numbers

For real numbers that cannot be exactly represented in machine precision (e.g., `0.1`, `1/3`, `pi`), use the `convert` function to make them consistent with the type of the function input. For example,
```julia
# Assume we are handling `pi` in function
function foo(..., input, ...)
  RealT = eltype(input) # see **notes** below
  # ...
  c1 = convert(RealT, pi) * c2 # sample operation
  # ...
end
```

### Integer numbers

Integers need special consideration. Using functions like `convert` (as mentioned above), `zero`, and `one` to change integers to a specific type or keeping them in integer format is feasible in most cases. We aim to balance code readability and consistency while maintaining type stability. Here are some examples to guide our developers.
```julia
# The first example - `SVector`, keep code consistency within `SVector`
SVector(0, 0, 0)
SVector(zero(RealT), zero(RealT), zero(RealT))
Svector(one(RealT), one(RealT), one(RealT))

# The second example - inner functions, keep them type-stable as well
function foo(..., input, ...)
RealT = eltype(input) # see **notes** below
# ...
c1 = c2 > 0.5f0 ? one(RealT) : convert(RealT, 0.1) # make type-stable
# ...
end

# The third example - some operations (e.g., `/`, `sqrt`, `inv`), convert them definitely
c1 = convert(RealT, 4) # suppose we get RealT before
c2 = 1 / c1
c3 = sqrt(c1)
c4 = inv(c1)
```
In general, in the case of integer numbers, our developers should apply a case-by-case strategy to maintain type stability.

### Notes
1. If the function gets a local pointwise vector of the solution variables `u` such as `flux(u, equations)`, use `u` to determine the real type `eltype(u)`.
2. If `u` is not passed as an argument but a vector of coordinates `x` such as `initial_condition(x, t, equations)`, use `eltype(x)` instead.
3. Choose an appropriate argument to determine the real type otherwise.
