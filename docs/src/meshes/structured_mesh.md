# Structured mesh

The [`StructuredMesh`](@ref) is a structured, curvilinear, conforming
mesh type available for one-, two-, and three-dimensional simulations.
An application of the [`StructuredMesh`](@ref) using a user-defined mapping 
is provided by [one of the tutorials](https://trixi-framework.github.io/Trixi.jl/stable/tutorials/structured_mesh_mapping/).

Due to its curvilinear nature, (numerical) fluxes need to implement methods
dispatching on the `normal::AbstractVector`. Rotationally invariant equations
such as the compressible Euler equations can use [`FluxRotated`](@ref) to
wrap numerical fluxes implemented only for Cartesian meshes. This simplifies
the re-use of existing functionality for the [`TreeMesh`](@ref) but is usually
less efficient, cf. [PR #550](https://github.com/trixi-framework/Trixi.jl/pull/550).
