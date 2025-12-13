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
    flux_parabolic(u_ll, u_rr, gradient_or_divergence, mesh, equations,
                   parabolic_scheme::ViscousFormulationBassiRebay1)

This computes the classical BR1 flux. Since the interface flux for both the 
DG gradient and DG divergence under BR1 are identical, this function does 
not need to be specialized for `Gradient` and `Divergence`.
"""
function flux_parabolic(u_ll, u_rr, gradient_or_divergence, mesh, equations,
                        parabolic_scheme::ViscousFormulationBassiRebay1)
    return 0.5f0 * (u_ll + u_rr)
end

"""
    ViscousFormulationLocalDG(penalty_parameter)

The local DG (LDG) flux from "The Local Discontinuous Galerkin Method for Time-Dependent
Convection-Diffusion Systems" by Cockburn and Shu (1998).

The parabolic "upwinding" vector is currently implemented for `TreeMesh`; for all other mesh types,
the LDG solver is equivalent to [`ViscousFormulationBassiRebay1`](@ref) with an LDG-type penalization.

- Cockburn and Shu (1998).
  The Local Discontinuous Galerkin Method for Time-Dependent
  Convection-Diffusion Systems
  [DOI: 10.1137/S0036142997316712](https://doi.org/10.1137/S0036142997316712)
"""
struct ViscousFormulationLocalDG{P}
    penalty_parameter::P
end

"""
    ViscousFormulationLocalDG()

The minimum dissipation local DG (LDG) flux from "An Analysis of the Minimal Dissipation Local 
Discontinuous Galerkin Method for Convection–Diffusion Problems" by Cockburn and Dong (2007). 
This scheme corresponds to an LDG parabolic "upwinding/downwinding" but no LDG penalty parameter. 
Cockburn and Dong proved that this scheme is still stable despite the zero penalty parameter. 

- Cockburn and Dong (2007)  
  An Analysis of the Minimal Dissipation Local Discontinuous 
  Galerkin Method for Convection–Diffusion Problems.
  [DOI: 10.1007/s10915-007-9130-3](https://doi.org/10.1007/s10915-007-9130-3)
"""
ViscousFormulationLocalDG() = ViscousFormulationLocalDG(nothing)

"""
    flux_parabolic(u_ll, u_rr, ::Gradient, mesh::TreeMesh, equations,
                   parabolic_scheme::ViscousFormulationLocalDG)

    flux_parabolic(u_ll, u_rr, ::Divergence, mesh::TreeMesh, equations,
                   parabolic_scheme::ViscousFormulationLocalDG)

These fluxes computes the gradient and divergence interface fluxes for the 
local DG method. The local DG method uses an "upwind/downwind" flux for the 
gradient and divergence (i.e., if the gradient is upwinded, the divergence
must be downwinded in order to preserve symmetry and positive definiteness). 
"""
function flux_parabolic(u_ll, u_rr, ::Gradient, mesh::TreeMesh, equations,
                        parabolic_scheme::ViscousFormulationLocalDG)
    # The LDG flux is {{f}} + beta * [[f]], where beta is the LDG "switch", 
    # which we set to -1 on the left and +1 on the right in 1D. The sign of the 
    # jump term should be opposite that of the sign used in the divergence flux. 
    # This is equivalent to setting the flux equal to `u_ll` for the gradient,
    # and `u_rr` for the divergence. 
    return u_ll # Use the upwind value for the gradient interface flux
end

function flux_parabolic(u_ll, u_rr, ::Divergence, mesh::TreeMesh, equations,
                        parabolic_scheme::ViscousFormulationLocalDG)
    return u_rr # Use the downwind value for the divergence interface flux
end

default_parabolic_solver() = ViscousFormulationBassiRebay1()

function default_artificial_viscosity(equations)

  # set diffusivity equal to 1, but scale in rhs
  if ndims(equations)==1
    return LaplaceDiffusionEntropyVariables1D(1, equations)
  elseif ndims(equations)==2
    return LaplaceDiffusionEntropyVariables2D(1, equations)
  else # if ndims(equations)==3
    return LaplaceDiffusionEntropyVariables3D(1, equations)        
  end
end