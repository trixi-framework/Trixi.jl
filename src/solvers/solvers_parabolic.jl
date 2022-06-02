"""
  ViscousFluxBassiRebay1()

The classical BR1 flux from

- F. Bassi, S. Rebay (1997)
  A High-Order Accurate Discontinuous Finite Element Method for
  the Numerical Solution of the Compressible Navier-Stokes Equations
  [DOI: 10.1006/jcph.1996.5572](https://doi.org/10.1006/jcph.1996.5572)
"""
struct ViscousFluxBassiRebay1 end

# no penalization for a BR1 parabolic solver
function calc_viscous_penalty!(scalar_flux_face_values, u_face_values, t, boundary_conditions,
                               mesh, equations::AbstractEquationsParabolic, dg::DGMulti,
                               dg_parabolic::ViscousFluxBassiRebay1, cache, cache_parabolic)
  return nothing
end

"""
    ViscousFluxLocalDG(penalty_parameter)

The local DG (LDG) flux from "The Local Discontinuous Galerkin Method for Time-Dependent
Convection-Diffusion Systems" by Cockburn and Shu (1998).

Note that, since this implementation does not involve the parabolic "upwinding" vector,
the LDG solver is equivalent to [`ViscousFluxBassiRebay1`](@ref) with an LDG-type penalization.

- Cockburn and Shu (1998).
  The Local Discontinuous Galerkin Method for Time-Dependent
  Convection-Diffusion Systems
  [DOI: 10.1137/S0036142997316712](https://doi.org/10.1137/S0036142997316712)
"""
struct ViscousFluxLocalDG{P}
  penalty_parameter::P
end

default_parabolic_solver() = ViscousFluxBassiRebay1()