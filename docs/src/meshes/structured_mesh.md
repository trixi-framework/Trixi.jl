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
For [`StructuredMesh`](@ref)es, boundary conditions are defined and stored in [named tuples](https://docs.julialang.org/en/v1/manual/functions/#Named-Tuples) (see, for example, `examples/structured_1d_dgsem/elixir_euler_source_terms_nonperiodic.jl`).  
If you want to apply the same boundary condition to all faces of the mesh, you can use the `boundary_condition_default(mesh, boundary_condition)` function, as demonstrated in `examples/structured_1d_dgsem/elixir_euler_source_terms_nonperiodic.jl`, `examples/structured_2d_dgsem/elixir_euler_rayleigh_taylor_instability.jl` and `examples/structured_3d_dgsem/elixir_euler_source_terms_nonperiodic_curved.jl`.
