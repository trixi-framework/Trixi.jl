"""
  struct BR1 end

The Bassi-Rebay flux from "A High-Order Accurate Discontinuous Finite Element Method for
the Numerical Solution of the Compressible Navier-Stokes Equations" by Bassi, Rebay (1997).

https://doi.org/10.1006/jcph.1996.5572
"""
struct BR1 end

# no penalization for a BR1 parabolic solver
function calc_viscous_penalty!(scalar_flux_face_values, u_face_values, t, boundary_conditions,
                               mesh, equations::AbstractEquationsParabolic, dg::DGMulti,
                               dg_parabolic::BR1, cache, cache_parabolic)
  return nothing
end

"""
  struct LDG{P}
    penalty_parameter::P
  end

The local DG (LDG) flux from "The Local Discontinuous Galerkin Method for Time-Dependent
Convection-Diffusion Systems" by Cockburn and Shu (1998).

Note that, since this implementation does not involve the parabolic "upwinding" vector,
the LDG solver is equivalent to BR1 with an LDG-type penalization term.

https://doi.org/10.1137/S0036142997316712
"""
struct LDG{P}
  penalty_parameter::P
end

default_parabolic_solver() = BR1()