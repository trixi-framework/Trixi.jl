# Overview of the structure of Trixi

Trixi is designed as a library of components for discretizations of hyperbolic
conservation laws. Thus, it is not a monolithic PDE solver that is configured at runtime
via parameter files, as it is often found in classical numerical simulation codes.
Instead, each simulation is configured by pure Julia code. Many examples of such
simulation setups, called *elixirs* in Trixi, are provided in the
[examples](https://github.com/trixi-framework/Trixi.jl/blob/main/examples)
folder.

Trixi uses the method of lines, i.e., the full space-time discretization is separated into two steps;
the spatial semidiscretization is performed at first and the resulting ODE system is solved numerically
using a suitable time integration method.
Thus, the main ingredients of an elixir designed
to solve a PDE numerically are the spatial semidiscretization and the time
integration scheme.


## Semidiscretizations

Semidiscretizations are high-level descriptions of spatial discretizations
specialized for certain PDEs. Trixi's main focus is on hyperbolic conservation
laws represented in a [`SemidiscretizationHyperbolic`](@ref).
Such semidiscretizations are usually named `semi` in Trixi.

![semidiscretization_overview](https://user-images.githubusercontent.com/12693098/124783641-83171e80-df45-11eb-8757-daac80cd1599.png)

The basic building blocks of a semidiscretization are

- a `mesh` describing the geometry of the domain
- a set of `equations` describing the physical model
- a `solver` describing the numerical approach

In addition, a semidiscretization bundles initial and boundary conditions, and
possible source terms. These different ingredients of a semidiscretization can
be configured individually and combined together.
When a semidiscretization is constructed, it will create an internal `cache`,
i.e., a collection of setup-specific data structures,
that is usually passed to all lower level functions.

Due to Trixi's modular nature using Julia's multiple dispatch features, new
ingredients can be created specifically for a certain combination of other
ingredients. For example, a new `mesh` type can be created and implemented at
first only for a specific solver. Thus, there is no need to consider all
possible combinations of `mesh`es, `equations`, and `solver`s when implementing
new features. This allows rapid prototyping of new ideas and is one of the main
design goals behind Trixi. Below is a brief overview of the availability of
different features on different mesh types.

| Feature                                                      | [`TreeMesh`](@ref) | [`StructuredMesh`](@ref) | [`UnstructuredMesh2D`](@ref) | [`P4estMesh`](@ref) | [`VertexMappedMesh`](@ref) | Further reading
|:-------------------------------------------------------------|:------------------:|:------------------------:|:----------------------------:|:-------------------:|:--------------------------:|:-----------------------------------------
| Spatial dimension                                            |     1D, 2D, 3D     |        1D, 2D, 3D        |              2D              |        2D, 3D       |          1D, 2D, 3D        |
| Coordinates                                                  |      Cartesian     |        curvilinear       |          curvilinear         |     curvilinear     |            affine          |
| Connectivity                                                 |  *h*-nonconforming |        conforming        |          conforming          |  *h*-nonconforming  |          conforming        |
| Element type                                                 | line, square, cube |     line, quadᵃ, hexᵃ    |             quadᵃ            |     quadᵃ, hexᵃ     |    simplex, quadᵃ, hexᵃ    |
| Adaptive mesh refinement                                     |          ✅         |             ❌            |               ❌              |          ✅          |               ❌            | [`AMRCallback`](@ref)
| Solver type                                                  |   [`DGSEM`](@ref)  |      [`DGSEM`](@ref)     |        [`DGSEM`](@ref)       |   [`DGSEM`](@ref)   |       [`DGMulti`](@ref)    |
| Domain                                                       |      hypercube     |     mapped hypercube     |           arbitrary          |      arbitrary      |       arbitrary (affine)   |
| Weak form                                                    |          ✅         |             ✅            |               ✅              |          ✅          |               ✅            | [`VolumeIntegralWeakForm`](@ref)
| Flux differencing                                            |          ✅         |             ✅            |               ✅              |          ✅          |               ✅            | [`VolumeIntegralFluxDifferencing`](@ref)
| Shock capturing                                              |          ✅         |             ✅            |               ✅              |          ✅          |               ❌            | [`VolumeIntegralShockCapturingHG`](@ref)
| Nonconservative equations                                    |          ✅         |             ✅            |               ✅              |          ✅          |               ✅            | e.g., GLM MHD or shallow water equations

ᵃ: quad = quadrilateral, hex = hexahedron

## Time integration methods

Trixi is compatible with the [SciML ecosystem for ordinary differential equations](https://diffeq.sciml.ai/latest/).
In particular, a spatial semidiscretization can be wrapped in an ODE problem
using [`semidiscretize`](@ref), which returns an `ODEProblem`. This `ODEProblem` is a wrapper
of `Trixi.rhs!(du_ode, u_ode, semi, t)`, which gets called in ODE solvers.
Further information can be found in the
[section on time integration methods](@ref time-integration).


## Next steps

We explicitly encourage people interested in Trixi to have a look at the
[examples](https://github.com/trixi-framework/Trixi.jl/blob/main/examples)
bundled with Trixi to get an impression of what is possible and the general
look and feel of Trixi.
Before doing that, it is usually good to get an idea of
[how to visualize numerical results](@ref visualization).

If you like learning by doing, looking at the tutorials and trying to mix
your own elixirs based thereon is probably a good next step.
Otherwise, you can further dig into the documentation by looking at Trixi's basic building blocks.
