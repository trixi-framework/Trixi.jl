# Conventions

## Spatial dimensions and directions

We use the following numbering schemes on Cartesian meshes.
- The `orientation`s are numbered as
  `1 => x, 2 => y, 3 => z`.
  For example, numerical fluxes such as
  `flux_central(u_ll, u_rr, orientation, equations::AbstractEquations)`
  use the `orientation` in this way.
- The `direction`s are numbered as
  `1 => -x, 2 => +x, 3 => -y, 4 => +y, 5 => -z, 6 => +z`.
  For example, the `boundary_condition`s are ordered in this way
  when a `Tuple` of boundary conditions per direction is passed
  to the constructor of a `SemidiscretizationHyperbolic`.
