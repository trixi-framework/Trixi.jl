# Tree mesh

The [`TreeMesh`](@ref) is a Cartesian, $h$-non-conforming mesh type
used in many parts of Trixi.jl. Often, the support for this mesh type is
developed best since it was the first mesh type in Trixi.jl,
and it is available in one, two, and three space dimensions.

It is limited to hypercube domains (that is, lines in 1D, squares in 2D and cubes in 3D) but supports AMR via the [`AMRCallback`](@ref).
Due to its Cartesian nature, (numerical) fluxes need to implement methods
dispatching on the `orientation::Integer` as described in the
[conventions](@ref conventions).


### Boundary conditions
For [`TreeMesh`](@ref)es, boundary conditions are defined and stored in named tuples (see, for example, `examples/tree_1d_dgsem/elixir_euler_source_terms_nonperiodic.jl`).  
If you want to apply the same boundary condition to all faces of the mesh, you can use the `boundary_condition_default(mesh, boundary_condition)` function, as demonstrated in `examples/tree_1d_dgsem/elixir_euler_source_terms_nonperiodic.jl`, `examples/tree_2d_dgsem/elixir_euler_source_terms_nonperiodic.jl` and `examples/tree_3d_dgsem/elixir_advection_extended.jl`.