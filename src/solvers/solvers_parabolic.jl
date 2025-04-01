"""
  ViscousFormulationBassiRebay1()

The classical BR1 flux from

- F. Bassi, S. Rebay (1997)
  A High-Order Accurate Discontinuous Finite Element Method for
  the Numerical Solution of the Compressible Navier-Stokes Equations
  [DOI: 10.1006/jcph.1996.5572](https://doi.org/10.1006/jcph.1996.5572)

A more detailed study of the BR1 scheme for the DGSEM can be found in
- G. J. Gassner, A. R. Winters, F. J. Hindenlang, D. Kopriva (2018)
  The BR1 Scheme is Stable for the Compressible Navier-Stokes Equations
  [DOI: 10.1007/s10915-018-0702-1](https://doi.org/10.1007/s10915-018-0702-1)

The BR1 scheme works well for convection-dominated problems, but may cause instabilities or 
reduced convergence for diffusion-dominated problems. 
In the latter case, the [`ViscousFormulationLocalDG`](@ref) scheme is recommended.
"""
struct ViscousFormulationBassiRebay1 end

"""
    ViscousFormulationLocalDG(penalty_parameter)

The local DG (LDG) flux from "The Local Discontinuous Galerkin Method for Time-Dependent
Convection-Diffusion Systems" by Cockburn and Shu (1998).

Note that, since this implementation does not involve the parabolic "upwinding" vector,
the LDG solver is equivalent to [`ViscousFormulationBassiRebay1`](@ref) with an LDG-type penalization.

- Cockburn and Shu (1998).
  The Local Discontinuous Galerkin Method for Time-Dependent
  Convection-Diffusion Systems
  [DOI: 10.1137/S0036142997316712](https://doi.org/10.1137/S0036142997316712)
"""
struct ViscousFormulationLocalDG{P}
    penalty_parameter::P
end

default_parabolic_solver() = ViscousFormulationBassiRebay1()
