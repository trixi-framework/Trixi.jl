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


## Cells vs. elements vs. nodes

To uniquely distinguish between different components of the discretization, we use the
following naming conventions:
* The computational domain is discretized by a *mesh*, which is made up of
  individual **cells**. In general, neither the mesh nor the cells should be
  aware of any solver-specific knowledge, i.e., they should not store anything
  that goes beyond the geometrical information and the connectivity.
* The numerical *solvers* do not directly store their information inside the mesh,
  but use own data structures. Specifically, for each mesh cell on which
  a solver wants to operate, the solver creates an **element** to store
  solver-specific data.
* For discretization schemes such as the discontinuous Galerkin or the finite
  element method, inside each element multiple **nodes** may be defined, which
  hold nodal information. The nodes are again a solver-specific component, just
  like the elements.


## Variable names

- Use descriptive names (using `snake_case` for variables/functions and `CamelCase` for types)
- Use a suffix `_` as in `name_` for local variables that would otherwise hide existing symbols.
- Use a prefix `_` as in `_name` to indicate internal methods/data that are "fragile" in the
  sense that there's no guarantee that they might get changed without notice. These are also not
  documented with a docstring (but maybe with comments using `#`).
