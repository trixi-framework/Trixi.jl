# P4est-based mesh

The [`P4estMesh`](@ref) is an unstructured, curvilinear, nonconforming
mesh type with the possibility for adaptive mesh refinement (AMR) based on
the C library [p4est](https://github.com/cburstedde/p4est). See
[`AMRCallback`](@ref) for further information.

Due to its curvilinear nature, (numerical) fluxes need to implement methods
dispatching on the `normal::AbstractVector`. Rotationally invariant equations
such as the compressible Euler equations can use [`FluxRotated`](@ref) to
wrap numerical fluxes implemented only for Cartesian meshes. This simplifies
the re-use of existing functionality for the [`TreeMesh`](@ref) but is usually
less efficient, cf. [PR #550](https://github.com/trixi-framework/Trixi.jl/pull/550).

