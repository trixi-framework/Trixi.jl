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
For [`TreeMesh`](@ref)es, boundary conditions are defined and stored in named tuples (as shown  for example in `examples/tree_2d_dgsem/elixir_advection_diffusion_nonperiodic.jl`). If you’d like to apply the same condition to every face of the mesh, you can use the convenient `functions boundary_condition_default_tree_1D`, `boundary_condition_default_tree_2D` and `boundary_condition_default_tree_3D`. For example, in the two dimensional case:

```julia
initial_condition = initial_condition_eriksson_johnson

boundary_conditions =  boundary_condition_default_tree_2D(BoundaryConditionDirichlet(initial_condition))
```