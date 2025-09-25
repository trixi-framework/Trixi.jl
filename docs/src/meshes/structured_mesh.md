# Structured mesh

The [`StructuredMesh`](@ref) is a structured, curvilinear, conforming
mesh type available for one-, two-, and three-dimensional simulations.
An application of the [`StructuredMesh`](@ref) using a user-defined mapping 
is provided by [one of the tutorials](https://trixi-framework.github.io/TrixiDocumentation/stable/tutorials/structured_mesh_mapping/).

Due to its curvilinear nature, (numerical) fluxes need to implement methods
dispatching on the `normal::AbstractVector`. Rotationally invariant equations
such as the compressible Euler equations can use [`FluxRotated`](@ref) to
wrap numerical fluxes implemented only for Cartesian meshes. This simplifies
the re-use of existing functionality for the [`TreeMesh`](@ref) but is usually
less efficient, cf. [PR #550](https://github.com/trixi-framework/Trixi.jl/pull/550).

### Boundary conditions
For [`StructuredMesh`](@ref)es, boundary conditions are defined and stored in named tuples (as shown for example in `examples/structured_1d_dgsem/elixir_euler_source_terms_nonperiodic.jl`). If you’d like to apply the same condition to every face of the mesh, you can use the convenient functions `boundary_condition_default_structured_1D`, `boundary_condition_default_structured_2D`  and `boundary_condition_default_structured_3D`. For example, in the one dimensional case:

```julia
boundary_condition = BoundaryConditionDirichlet(initial_condition)
boundary_conditions = boundary_condition_default_structured_1D(boundary_condition)
```
