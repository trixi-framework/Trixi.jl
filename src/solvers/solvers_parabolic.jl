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
    flux_parabolic(f_ll, f_rr,
                   gradient_or_divergence, equations_parabolic,
                   parabolic_scheme::ViscousFormulationBassiRebay1)

    flux_parabolic(f_ll, f_rr, normal_direction::AbstractVector,
                   gradient_or_divergence, equations_parabolic,
                   parabolic_scheme::ViscousFormulationBassiRebay1)

This computes the classical BR1 flux. Since the interface flux for both the 
DG gradient and DG divergence under BR1 are identical, this function does 
not need to be specialized for `Gradient` and `Divergence`.

`normal_direction` is not used in the BR1 flux,
but is included as an argument for consistency with the [`ViscousFormulationLocalDG`](@ref) flux,
which does use the `normal_direction` to compute the LDG "switch" on the generally non-Cartesian [`P4estMesh`](@ref).
"""
function flux_parabolic(f_ll, f_rr,
                        gradient_or_divergence, equations_parabolic,
                        parabolic_scheme::ViscousFormulationBassiRebay1)
    return 0.5f0 * (f_ll + f_rr)
end
# For `P4estMesh`
function flux_parabolic(f_ll, f_rr, normal_direction::AbstractVector,
                        gradient_or_divergence, equations_parabolic,
                        parabolic_scheme::ViscousFormulationBassiRebay1)
    return 0.5f0 * (f_ll + f_rr)
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

@doc raw"""
    flux_parabolic(f_ll, f_rr,
                   ::Gradient, equations_parabolic,
                   parabolic_scheme::ViscousFormulationLocalDG)

    flux_parabolic(f_ll, f_rr, normal_direction,
                   ::Gradient, equations_parabolic,
                   parabolic_scheme::ViscousFormulationLocalDG)

These fluxes computes the gradient and divergence interface fluxes for the 
local DG method. The local DG method uses an "upwind/downwind" flux for the 
gradient and divergence (i.e., if the gradient is upwinded, the divergence
must be downwinded in order to preserve symmetry and positive definiteness).
Here, we use the convention that the gradient flux is upwinded, thus we have
```math
f_{\text{gradient}} = f_L
```
on the Cartesian [`TreeMesh`](@ref).

For the [`P4estMesh`](@ref), the `normal_direction` is used to compute the LDG "switch" for the upwinding/downwinding.
This is realized by taking the sign of the dot product of the normal and positive-coordinate direction vector:
```math
s = \text{sign}(\vec{n} \cdot \vec{1}
f = \frac{1}{2}\big(f_L + f_R - s (f_R - f_L)\big)
```
"""
function flux_parabolic(f_ll, f_rr,
                        ::Gradient, equations_parabolic,
                        parabolic_scheme::ViscousFormulationLocalDG)
    # The LDG flux is {{f}} + beta * [[f]], where beta is the LDG "switch", 
    # which we set to -1 on the left and +1 on the right in 1D. The sign of the 
    # jump term should be opposite that of the sign used in the divergence flux. 
    # This is equivalent to setting the flux equal to `f_ll` for the gradient,
    # and `f_rr` for the divergence. 
    return f_ll # Use the upwind value for the gradient interface flux
end

function flux_parabolic(f_ll, f_rr, normal_direction,
                        ::Gradient, equations_parabolic,
                        parabolic_scheme::ViscousFormulationLocalDG)
    ldg_switch = sign(sum(normal_direction)) # equivalent to sign(dot(normal_direction, ones))
    return 0.5f0 * (f_ll + f_rr - ldg_switch * (f_rr - f_ll))
end

@doc raw"""
    flux_parabolic(f_ll, f_rr,
                   ::Divergence, equations_parabolic,
                   parabolic_scheme::ViscousFormulationLocalDG)

    flux_parabolic(f_ll, f_rr, normal_direction,
                   ::Divergence, equations_parabolic,  
                   parabolic_scheme::ViscousFormulationLocalDG)

These fluxes computes the gradient and divergence interface fluxes for the 
local DG method. The local DG method uses an "upwind/downwind" flux for the 
gradient and divergence (i.e., if the gradient is upwinded, the divergence
must be downwinded in order to preserve symmetry and positive definiteness).
Here, we use the convention that the divergence flux is upwinded, thus we have
```math
f_{\text{divergence}} = f_R
```
on the Cartesian [`TreeMesh`](@ref).

For the [`P4estMesh`](@ref), the `normal_direction` is used to compute the LDG "switch" for the upwinding/downwinding.
This is realized by taking the sign of the dot product of the normal and positive-coordinate direction vector:
```math
s = \text{sign}(\vec{n} \cdot \vec{1}
f = \frac{1}{2}\big(f_L + f_R + s (f_R - f_L)\big)
```
"""
function flux_parabolic(f_ll, f_rr, ::Divergence, equations_parabolic,
                        parabolic_scheme::ViscousFormulationLocalDG)
    return f_rr # Use the downwind value for the divergence interface flux
end

function flux_parabolic(f_ll, f_rr, normal_direction,
                        ::Divergence, equations_parabolic,
                        parabolic_scheme::ViscousFormulationLocalDG)
    ldg_switch = sign(sum(normal_direction)) # equivalent to sign(dot(normal_direction, ones))
    return 0.5f0 * (f_ll + f_rr + ldg_switch * (f_rr - f_ll))
end

default_parabolic_solver() = ViscousFormulationBassiRebay1()
