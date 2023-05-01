# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin

@doc raw"""
    ShallowWaterTwoLayerEquations1D(gravity, H0, rho_upper, rho_lower)

Two-Layer Shallow Water equations (2LSWE) in one space dimension. The equations are given by
```math
\begin{alignat*}{4}
&\frac{\partial}{\partial t}h_{upper} 
&&+ \frac{\partial}{\partial x}\left(h_{upper} v_{1,upper}\right) 
&&= 0 \\
&\frac{\partial}{\partial t}\left(h_{upper}v_{1,upper}\right) 
&&+ \frac{\partial}{\partial x}\left(h_{upper}v_{1,upper}^2 + \dfrac{gh_{upper}^2}{2}\right) 
&&= -gh_{upper}\frac{\partial}{\partial x}\left(b+h_{lower}\right)\\
&\frac{\partial}{\partial t}h_{lower}  
&&+ \frac{\partial}{\partial x}\left(h_{lower}v_{1,lower}\right) 
&&= 0 \\
&\frac{\partial}{\partial t}\left(h_{lower}v_{1,lower}\right)  
&&+ \frac{\partial}{\partial x}\left(h_{lower}v_{1,lower}^2 + \dfrac{gh_{lower}^2}{2}\right) 
&&= -gh_{lower}\frac{\partial}{\partial x}\left(b+\dfrac{\rho_{upper}}{\rho_{lower}}h_{upper}\right).
\end{alignat*}
```
The unknown quantities of the 2LSWE are the water heights of the {lower} layer ``h_{lower}`` and the 
{upper} layer ``h_{upper}`` with respective velocities ``v_{1,upper}`` and ``v_{1,lower}``. The gravitational constant is 
denoted by `g`, the layer densitites by ``\rho_{upper}``and ``\rho_{lower}`` and the (possibly) variable 
bottom topography function ``b(x)``. The conservative variable water height ``h_{lower}`` is measured 
from the bottom topography ``b`` and ``h_{upper}`` relative to ``h_{lower}``, therefore one also defines the 
total water heights as ``H_{upper} = h_{upper} + h_{upper} + b`` and ``H_{lower} = h_{lower} + b``.

The densities must be chosen such that ``\rho_{upper} < \rho_{lower}``, to make sure that the heavier fluid 
``\rho_{lower}`` is in the bottom layer and the lighter fluid ``\rho_{upper}`` in the {upper} layer.

The additional quantity ``H_0`` is also available to store a reference value for the total water
height that is useful to set initial conditions or test the "lake-at-rest" well-balancedness.

The bottom topography function ``b(x)`` is set inside the initial condition routine
for a particular problem setup.

In addition to the unknowns, Trixi currently stores the bottom topography values at the 
approximation points despite being fixed in time. This is done for convenience of computing the 
bottom topography gradients on the fly during the approximation as well as computing auxiliary 
quantities like the total water height ``H`` or the entropy variables.
This affects the implementation and use of these equations in various ways:
* The flux values corresponding to the bottom topography must be zero.
* The bottom topography values must be included when defining initial conditions, boundary 
  conditions or source terms.
* [`AnalysisCallback`](@ref) analyzes this variable.
* Trixi's visualization tools will visualize the bottom topography by default.

A good introduction for the 2LSWE is available in Chapter 12 of the book:
- Benoit Cushman-Roisin (2011)\
  Introduction to geophyiscal fluid dynamics: physical and numerical aspects\
  <https://www.sciencedirect.com/bookseries/international-geophysics/vol/101/suppl/C>\
  ISBN: 978-0-12-088759-0
"""
struct ShallowWaterTwoLayerEquations1D{RealT<:Real} <: AbstractShallowWaterEquations{1,5}
  gravity::RealT   # gravitational constant
  H0::RealT        # constant "lake-at-rest" total water height
  rho_upper::RealT # lower layer density
  rho_lower::RealT # upper layer density
  r::RealT         # ratio of rho_upper / rho_lower
end

# Allow for flexibility to set the gravitational constant within an elixir depending on the
# application where `gravity_constant=1.0` or `gravity_constant=9.81` are common values.
# The reference total water height H0 defaults to 0.0 but is used for the "lake-at-rest"
# well-balancedness test cases. Densities must be specified such that rho_upper <= rho_lower.
function ShallowWaterTwoLayerEquations1D(; gravity_constant, H0=zero(gravity_constant), rho_upper, rho_lower)
  # Assign density ratio if rho_upper <= rho_lower
  if rho_upper > rho_lower
    error("Invalid input: Densities must be chosen such that rho_upper <= rho_lower")
  else
    r = rho_upper / rho_lower
  end
  ShallowWaterTwoLayerEquations1D(gravity_constant, H0, rho_upper, rho_lower, r)
end

have_nonconservative_terms(::ShallowWaterTwoLayerEquations1D) = True()
varnames(::typeof(cons2cons), ::ShallowWaterTwoLayerEquations1D) = ("h_upper", "h_v1_upper",
                                                                    "h_lower", "h_v1_lower", "b")
# Note, we use the total water height, H_lower = h_upper + h_lower + b, and first layer total height
# H_upper = h_upper + b as the first primitive variable for easier visualization and setting initial
# conditions
varnames(::typeof(cons2prim), ::ShallowWaterTwoLayerEquations1D) = ("H_upper", "v1_upper", 
                                                                    "H_lower", "v1_lower", "b")


# Set initial conditions at physical location `x` for time `t`
"""
    initial_condition_convergence_test(x, t, equations::ShallowWaterTwoLayerEquations1D)

A smooth initial condition used for convergence tests in combination with
[`source_terms_convergence_test`](@ref) (and 
[`BoundaryConditionDirichlet(initial_condition_convergence_test)`](@ref) in non-periodic domains).
"""
function initial_condition_convergence_test(x, t, equations::ShallowWaterTwoLayerEquations1D)
  # some constants are chosen such that the function is periodic on the domain [0,sqrt(2)]
  ω = 2.0 * pi * sqrt(2.0)

  H_lower = 2.0 + 0.1 * sin(ω * x[1] + t)
  H_upper = 4.0 + 0.1 * cos(ω * x[1] + t)
  v1_lower = 1.0
  v1_upper = 0.9
  b  = 1.0 + 0.1 * cos(2.0 * ω * x[1])

  return prim2cons(SVector(H_upper, v1_upper, H_lower, v1_lower, b), equations)
end


"""
    source_terms_convergence_test(u, x, t, equations::ShallowWaterTwoLayerEquations1D)

Source terms used for convergence tests in combination with 
[`initial_condition_convergence_test`](@ref) 
(and [`BoundaryConditionDirichlet(initial_condition_convergence_test)`](@ref) 
in non-periodic domains).
"""
@inline function source_terms_convergence_test(u, x, t,
                                               equations::ShallowWaterTwoLayerEquations1D)
  # Same settings as in `initial_condition_convergence_test`. Some derivative simplify because
  # this manufactured solution velocity is taken to be constant
  ω = 2 * pi * sqrt(2.0)

  du1 = (-0.1*cos(t + ω*x[1]) - 0.1*sin(t + ω*x[1]) - 0.09*ω*cos(t + ω*x[1]) +
          - 0.09*ω*sin(t + ω*x[1]))
  du2 = (5.0 * (-0.1*ω*cos(t + ω*x[1]) - 0.1*ω*sin(t + ω*x[1])) * (4.0 + 0.2*cos(t + ω*x[1]) +
        -0.2*sin(t + ω*x[1])) + 0.1*ω*(20.0 + cos(t + ω*x[1]) - sin(t + ω*x[1])) * cos(t +
          ω*x[1]) - 0.09*cos(t + ω*x[1]) - 0.09*sin(t + ω*x[1]) - 0.081*ω*cos(t + ω*x[1]) +
        -0.081*ω*sin(t + ω*x[1]))
  du3 = 0.1*cos(t + ω*x[1]) + 0.1*ω*cos(t + ω*x[1]) + 0.2*ω*sin(2.0*ω*x[1])
  du4 = ((10.0 + sin(t + ω*x[1]) - cos(2ω*x[1]))*(-0.09*ω*cos(t + ω*x[1]) - 0.09*ω*sin(t +
          ω*x[1]) - 0.2*ω*sin(2*ω*x[1])) + 0.1*cos(t + ω*x[1]) + 0.1*ω*cos(t + ω*x[1]) +
          5.0 * (0.1*ω*cos(t + ω*x[1]) + 0.2*ω*sin(2.0*ω*x[1])) * (2.0 + 0.2*sin(t + ω*x[1]) +
        -0.2*cos(2.0*ω*x[1])) + 0.2*ω*sin(2.0*ω*x[1]))

  return SVector(du1, du2, du3, du4, zero(eltype(u)))
end


"""
    boundary_condition_slip_wall(u_inner, orientation_or_normal, x, t, surface_flux_function,
                                 equations::ShallowWaterTwoLayerEquations1D)

Create a boundary state by reflecting the normal velocity component and keep
the tangential velocity component unchanged. The boundary water height is taken from
the internal value.

For details see Section 9.2.5 of the book:
- Eleuterio F. Toro (2001)
  Shock-Capturing Methods for Free-Surface Shallow Flows  
  1st edition  
  ISBN 0471987662
"""
@inline function boundary_condition_slip_wall(u_inner, orientation_or_normal, direction,
                                              x, t, surface_flux_function,
                                              equations::ShallowWaterTwoLayerEquations1D)
  # create the "external" boundary solution state
  u_boundary = SVector(u_inner[1],
                      -u_inner[2],
                       u_inner[3],
                      -u_inner[4],
                       u_inner[5])

  # calculate the boundary flux
  if iseven(direction) # u_inner is "left" of boundary, u_boundary is "right" of boundary
    f = surface_flux_function(u_inner, u_boundary, orientation_or_normal, equations)
  else # u_boundary is "left" of boundary, u_inner is "right" of boundary
    f = surface_flux_function(u_boundary, u_inner, orientation_or_normal, equations)
  end
  return f
end


# Calculate 1D flux for a single point
# Note, the bottom topography has no flux
@inline function flux(u, orientation::Integer, equations::ShallowWaterTwoLayerEquations1D)
  h_upper, h_v1_upper, h_lower, h_v2_lower, _ = u

  # Calculate velocities
  v1_upper, v1_lower = velocity(u, equations)
  # Calculate pressure
  p1 = 0.5 * equations.gravity * h_upper^2
  p2 = 0.5 * equations.gravity * h_lower^2

  f1 = h_v1_upper
  f2 = h_v1_upper * v1_upper + p1
  f3 = h_v2_lower
  f4 = h_v2_lower * v1_lower + p2

  return SVector(f1, f2, f3, f4, zero(eltype(u)))
end


"""
    flux_nonconservative_wintermeyer_etal(u_ll, u_rr, orientation::Integer,
                                          equations::ShallowWaterTwoLayerEquations1D)

!!! warning "Experimental code"
    This numerical flux is experimental and may change in any future release.

Non-symmetric two-point volume flux discretizing the nonconservative (source) term
that contains the gradient of the bottom topography [`ShallowWaterTwoLayerEquations2D`](@ref) and an
additional term that couples the momentum of both layers. This is a slightly modified version 
to account for the additional source term compared to the standard SWE described in the paper.

Further details are available in the paper:
- Niklas Wintermeyer, Andrew R. Winters, Gregor J. Gassner and David A. Kopriva (2017)
  An entropy stable nodal discontinuous Galerkin method for the two dimensional
  shallow water equations on unstructured curvilinear meshes with discontinuous bathymetry
  [DOI: 10.1016/j.jcp.2017.03.036](https://doi.org/10.1016/j.jcp.2017.03.036)
"""
@inline function flux_nonconservative_wintermeyer_etal(u_ll, u_rr,
                                                       orientation::Integer,
                                                       equations::ShallowWaterTwoLayerEquations1D)
  # Pull the necessary left and right state information
  h_upper_ll, h_lower_ll = waterheight(u_ll, equations)
  h_upper_rr, h_lower_rr = waterheight(u_rr, equations)
  b_rr = u_rr[5]

  z = zero(eltype(u_ll))

  # Bottom gradient nonconservative term: (0, g*h_upper*(b+h_lower)_x, 
  #                                        0, g*h_lower*(b+r*h_upper)_x, 0)
  f = SVector(z,
              equations.gravity * h_upper_ll * (b_rr + h_lower_rr),
              z,
              equations.gravity * h_lower_ll * (b_rr + equations.r * h_upper_rr),
              z)
  return f
end


"""
    flux_nonconservative_fjordholm_etal(u_ll, u_rr, orientation::Integer,
                                        equations::ShallowWaterTwoLayerEquations1D)

!!! warning "Experimental code"
    This numerical flux is experimental and may change in any future release.
                                    
Non-symmetric two-point surface flux discretizing the nonconservative (source) term that contains 
the gradients of the bottom topography and an additional term that couples the momentum of both 
layers [`ShallowWaterTwoLayerEquations2D`](@ref).

Further details are available in the paper:
- Ulrik Skre Fjordholm (2012)
  Energy conservative and stable schemes for the two-layer shallow water equations.
  [DOI: 10.1142/9789814417099_0039](https://doi.org/10.1142/9789814417099_0039)
It should be noted that the equations are ordered differently and the
designation of the upper and lower layer has been changed which leads to a slightly different
formulation.
"""
@inline function flux_nonconservative_fjordholm_etal(u_ll, u_rr,
                                                     orientation::Integer,
                                                     equations::ShallowWaterTwoLayerEquations1D)
  # Pull the necessary left and right state information
  h_upper_ll, _, h_lower_ll, _, b_ll = u_ll
  h_upper_rr, _, h_lower_rr, _, b_rr = u_rr

  # Create average and jump values
  h_upper_average = 0.5 * (h_upper_ll + h_upper_rr)
  h_lower_average = 0.5 * (h_lower_ll + h_lower_rr)
  h_upper_jump = h_upper_rr - h_upper_ll
  h_lower_jump = h_lower_rr - h_lower_ll
  b_jump  = b_rr  - b_ll

  # Assign variables for constants for better readability
  g = equations.gravity

  z = zero(eltype(u_ll))

  # Bottom gradient nonconservative term: (0, g*h_upper*(b+h_lower)_x, 
  #                                        0, g*h_lower*(b+r*h_upper)_x, 0)
  f = SVector(
    z,
    g * h_upper_ll * (b_ll + h_lower_ll) + g * h_upper_average * (b_jump + h_lower_jump),
    z,
    g * h_lower_ll * (b_ll + equations.r * h_upper_ll) + g * h_lower_average * (b_jump +
        equations.r * h_upper_jump),
    z)
  return f
end


"""
    flux_fjordholm_etal(u_ll, u_rr, orientation,
                        equations::ShallowWaterTwoLayerEquations1D)

Total energy conservative (mathematical entropy for shallow water equations). When the bottom 
topography is nonzero this should only be used as a surface flux otherwise the scheme will not be 
well-balanced. For well-balancedness in the volume flux use [`flux_wintermeyer_etal`](@ref).

Details are available in Eq. (4.1) in the paper:
- Ulrik S. Fjordholm, Siddhartha Mishra and Eitan Tadmor (2011)
  Well-balanced and energy stable schemes for the shallow water equations with discontinuous 
  topography [DOI: 10.1016/j.jcp.2011.03.042](https://doi.org/10.1016/j.jcp.2011.03.042)
and the application to two layers is shown in the paper:
- Ulrik Skre Fjordholm (2012)
  Energy conservative and stable schemes for the two-layer shallow water equations.
  [DOI: 10.1142/9789814417099_0039](https://doi.org/10.1142/9789814417099_0039)
It should be noted that the equations are ordered differently and the
designation of the upper and lower layer has been changed which leads to a slightly different
formulation.
"""
@inline function flux_fjordholm_etal(u_ll, u_rr,
                                     orientation::Integer,
                                     equations::ShallowWaterTwoLayerEquations1D)
  # Unpack left and right state
  h_upper_ll, h_lower_ll = waterheight(u_ll, equations)
  v1_ll, v2_ll = velocity(u_ll, equations)
  h_upper_rr, h_lower_rr = waterheight(u_rr, equations)
  v1_rr, v2_rr = velocity(u_rr, equations)

  # Average each factor of products in flux
  h_upper_avg = 0.5 * (h_upper_ll + h_upper_rr)
  h_lower_avg = 0.5 * (h_lower_ll + h_lower_rr)
  v1_avg = 0.5 * (v1_ll + v1_rr)
  v2_avg = 0.5 * (v2_ll + v2_rr)
  p1_avg = 0.25 * equations.gravity * (h_upper_ll^2 + h_upper_rr^2)
  p2_avg = 0.25 * equations.gravity * (h_lower_ll^2 + h_lower_rr^2)

  # Calculate fluxes
  f1 = h_upper_avg * v1_avg
  f2 = f1 * v1_avg + p1_avg
  f3 = h_lower_avg * v2_avg
  f4 = f3 * v2_avg + p2_avg

  return SVector(f1, f2, f3, f4, zero(eltype(u_ll)))
end


"""
    flux_wintermeyer_etal(u_ll, u_rr, orientation,
                          equations::ShallowWaterTwoLayerEquations1D)
                      
Total energy conservative (mathematical entropy for two-layer shallow water equations) split form.
When the bottom topography is nonzero this scheme will be well-balanced when used as a `volume_flux`.
The `surface_flux` should still use, e.g., [`flux_fjordholm_etal`](@ref). To obtain the flux for the
two-layer shallow water equations the flux that is described in the paper for the normal shallow 
water equations is used within each layer.

Further details are available in Theorem 1 of the paper:
- Niklas Wintermeyer, Andrew R. Winters, Gregor J. Gassner and David A. Kopriva (2017)
  An entropy stable nodal discontinuous Galerkin method for the two dimensional
  shallow water equations on unstructured curvilinear meshes with discontinuous bathymetry
  [DOI: 10.1016/j.jcp.2017.03.036](https://doi.org/10.1016/j.jcp.2017.03.036)
"""
@inline function flux_wintermeyer_etal(u_ll, u_rr,
                                       orientation::Integer,
                                       equations::ShallowWaterTwoLayerEquations1D)
  # Unpack left and right state
  h_upper_ll, h_v1_upper_ll, h_lower_ll, h_v2_lower_ll, _ = u_ll
  h_upper_rr, h_v1_upper_rr, h_lower_rr, h_v2_lower_rr, _ = u_rr

  # Get the velocities on either side
  v1_ll, v2_ll = velocity(u_ll, equations)
  v1_rr, v2_rr = velocity(u_rr, equations)

  # Average each factor of products in flux
  v1_avg = 0.5 * (v1_ll + v1_rr)
  v2_avg = 0.5 * (v2_ll + v2_rr)
  p1_avg = 0.5 * equations.gravity * h_upper_ll * h_upper_rr
  p2_avg = 0.5 * equations.gravity * h_lower_ll * h_lower_rr

  # Calculate fluxes
  f1 = 0.5 * (h_v1_upper_ll + h_v1_upper_rr)
  f2 = f1 * v1_avg + p1_avg
  f3 = 0.5 * (h_v2_lower_ll + h_v2_lower_rr)
  f4 = f3 * v2_avg + p2_avg

  return SVector(f1, f2, f3, f4, zero(eltype(u_ll)))
end


"""
    flux_es_fjordholm_etal(u_ll, u_rr, orientation,
                           equations::ShallowWaterTwoLayerEquations1D)

Entropy stable surface flux for the two-layer shallow water equations. Uses the entropy 
conservative flux_fjordholm_etal and adds a Lax-Friedrichs type dissipation dependent on the jump 
of entropy variables.

Further details are available in the paper:
- Ulrik Skre Fjordholm (2012)
  Energy conservative and stable schemes for the two-layer shallow water equations.
  [DOI: 10.1142/9789814417099_0039](https://doi.org/10.1142/9789814417099_0039)
It should be noted that the equations are ordered differently and the
designation of the upper and lower layer has been changed which leads to a slightly different
formulation.
"""
@inline function flux_es_fjordholm_etal(u_ll, u_rr,
                                        orientation::Integer,
                                        equations::ShallowWaterTwoLayerEquations1D)
  # Compute entropy conservative flux but without the bottom topography
  f_ec = flux_fjordholm_etal(u_ll, u_rr,
                              orientation,
                              equations)

  # Get maximum signal velocity
  λ = max_abs_speed_naive(u_ll, u_rr, orientation, equations)
  # Get entropy variables but without the bottom topography
  q_rr = cons2entropy(u_rr,equations)
  q_ll = cons2entropy(u_ll,equations)

  # Average values from left and right
  u_avg = (u_ll + u_rr) / 2

  # Introduce variables for better readability
  rho_upper = equations.rho_upper
  rho_lower = equations.rho_lower
  g    = equations.gravity
  drho = rho_upper - rho_lower

  # Entropy Jacobian matrix
  H = @SMatrix [
    [-rho_lower/(g*rho_upper*drho);;
     -rho_lower*u_avg[2]/(g*rho_upper*u_avg[1]*drho);;
     1.0/(g*drho);;
     u_avg[4]/(g*u_avg[3]*drho);;
     0];
    [-rho_lower*u_avg[2]/(g*rho_upper*u_avg[1]*drho);;
     (g*rho_upper*u_avg[1]^3 - g*rho_lower*u_avg[1]^3 +
         -rho_lower*u_avg[2]^2)/(g*rho_upper*u_avg[1]^2*drho);;
     u_avg[2]/(g*u_avg[1]*drho);;
     u_avg[2]*u_avg[4]/(g*u_avg[1]*u_avg[3]*drho);;
     0];
    [1.0/(g*drho);;
     u_avg[2]/(g*u_avg[1]*drho);;
     -1.0/(g*drho);;
     -u_avg[4]/(g*u_avg[3]*drho);;
     0];
    [u_avg[4]/(g*u_avg[3]*drho);;
     u_avg[2]*u_avg[4]/(g*u_avg[1]*u_avg[3]*drho);;
     -u_avg[4]/(g*u_avg[3]*drho);;
     (g*rho_upper*u_avg[3]^3 - g*rho_lower*u_avg[3]^3 +
         -rho_lower*u_avg[4]^2)/(g*rho_lower*u_avg[3]^2*drho);;
     0];
    [0;;0;;0;;0;;0]]

  # Add dissipation to entropy conservative flux to obtain entropy stable flux
  f_es = f_ec - 0.5 * λ * H * (q_rr - q_ll)

  return SVector(f_es[1], f_es[2], f_es[3], f_es[4], zero(eltype(u_ll)))
end


# Calculate approximation for maximum wave speed for local Lax-Friedrichs-type dissipation as the
# maximum velocity magnitude plus the maximum speed of sound. This function uses approximate 
# eigenvalues using the speed of the barotropic mode as there is no simple way to calculate them 
# analytically.
# 
# A good overview of the derivation is given in:
# -  Jonas Nycander, Andrew McC. Hogg, Leela M. Frankcombe (2008)
#    Open boundary conditions for nonlinear channel Flows
#    [DOI: 10.1016/j.ocemod.2008.06.003](https://doi.org/10.1016/j.ocemod.2008.06.003)
@inline function max_abs_speed_naive(u_ll, u_rr,
                                     orientation::Integer,
                                     equations::ShallowWaterTwoLayerEquations1D)
  # Unpack left and right state
  h_upper_ll, h_v1_upper_ll, h_lower_ll, h_v2_lower_ll, _ = u_ll
  h_upper_rr, h_v1_upper_rr, h_lower_rr, h_v2_lower_rr, _ = u_rr

  # Get the averaged velocity
  v_m_ll = (h_v1_upper_ll + h_v2_lower_ll) / (h_upper_ll + h_lower_ll)
  v_m_rr = (h_v1_upper_rr + h_v2_lower_rr) / (h_upper_rr + h_lower_rr)

  # Calculate the wave celerity on the left and right
  h_upper_ll, h_lower_ll = waterheight(u_ll, equations)
  h_upper_rr, h_lower_rr = waterheight(u_rr, equations)
  c_ll = sqrt(equations.gravity * (h_upper_ll + h_lower_ll))
  c_rr = sqrt(equations.gravity * (h_upper_rr + h_lower_rr))

  return (max(abs(v_m_ll) + c_ll, abs(v_m_rr) + c_rr))
end


# Specialized `DissipationLocalLaxFriedrichs` to avoid spurious dissipation in the bottom 
# topography
@inline function (dissipation::DissipationLocalLaxFriedrichs)(u_ll, u_rr,
    orientation_or_normal_direction, equations::ShallowWaterTwoLayerEquations1D)
  λ = dissipation.max_abs_speed(u_ll, u_rr, orientation_or_normal_direction, equations)
  diss = -0.5 * λ * (u_rr - u_ll)
  return SVector(diss[1], diss[2], diss[3], diss[4], zero(eltype(u_ll)))
end


# Absolute speed of the barotropic mode
@inline function max_abs_speeds(u, equations::ShallowWaterTwoLayerEquations1D)
  h_upper, h_v1_upper, h_lower, h_v2_lower, _ = u
  
  # Calculate averaged velocity of both layers
  v_m = (h_v1_upper + h_v2_lower) / (h_upper + h_lower)
  c = sqrt(equations.gravity * (h_upper + h_lower))

  return (abs(v_m) + c)
end


# Helper function to extract the velocity vector from the conservative variables
@inline function velocity(u, equations::ShallowWaterTwoLayerEquations1D)
  h_upper, h_v1_upper, h_lower, h_v2_lower, _ = u

  v1_upper = h_v1_upper / h_upper
  v1_lower = h_v2_lower / h_lower
  return SVector(v1_upper, v1_lower)
end


# Convert conservative variables to primitive
@inline function cons2prim(u, equations::ShallowWaterTwoLayerEquations1D)
  h_upper, _, h_lower, _, b = u

  H_lower = h_lower + b
  H_upper = h_lower + h_upper + b
  v1_upper, v1_lower = velocity(u, equations)
  return SVector(H_upper, v1_upper, H_lower, v1_lower, b)
end


# Convert conservative variables to entropy variables
# Note, only the first four are the entropy variables, the fifth entry still just carries the 
# bottom topography values for convenience
@inline function cons2entropy(u, equations::ShallowWaterTwoLayerEquations1D)
  h_upper, _, h_lower, _, b = u
  v1_upper, v1_lower = velocity(u, equations)

  w1 = equations.rho_upper * (equations.gravity * (h_upper + h_lower + b) - 0.5 * v1_upper^2)
  w2 = equations.rho_upper * v1_upper
  w3 = equations.rho_lower * (equations.gravity * (equations.r * h_upper + h_lower + b) - 0.5 * v1_lower^2)
  w4 = equations.rho_lower * v1_lower
  return SVector(w1, w2, w3, w4, b)
end


# Convert primitive to conservative variables
@inline function prim2cons(prim, equations::ShallowWaterTwoLayerEquations1D)
  H_upper, v1_upper, H_lower, v1_lower, b = prim

  h_lower    = H_lower - b
  h_upper    = H_upper - h_lower - b
  h_v1_upper = h_upper * v1_upper
  h_v2_lower = h_lower * v1_lower
  return SVector(h_upper, h_v1_upper, h_lower, h_v2_lower, b)
end


@inline function waterheight(u, equations::ShallowWaterTwoLayerEquations1D)
  return SVector(u[1], u[3])
end


# Entropy function for the shallow water equations is the total energy
@inline entropy(cons, equations::ShallowWaterTwoLayerEquations1D) = energy_total(cons, equations)


# Calculate total energy for a conservative state `cons`
@inline function energy_total(cons, equations::ShallowWaterTwoLayerEquations1D)
  h_upper, h_lower, h_v1_upper, h_v2_lower, b = cons
  # Set new variables for better readability
  g = equations.gravity
  rho_upper = equations.rho_upper
  rho_lower = equations.rho_lower

  e = (0.5 * rho_upper * (h_v1_upper^2 / h_upper + g * h_upper^2) + 0.5 * rho_lower * (h_v2_lower^2 / h_lower + g * h_lower^2) +
      g * rho_lower * h_lower * b + g * rho_upper * h_upper * (h_lower + b))
  return e
end


# Calculate kinetic energy for a conservative state `cons`
@inline function energy_kinetic(u, equations::ShallowWaterTwoLayerEquations1D)
  h_upper, h_v1_upper, h_lower, h_v2_lower, _ = u
  return 0.5 * equations.rho_upper * h_v1_upper^2 / h_upper + 0.5 * equations.rho_lower * h_v2_lower^2 / h_lower
end


# Calculate potential energy for a conservative state `cons`
@inline function energy_internal(cons, equations::ShallowWaterTwoLayerEquations1D)
  return energy_total(cons, equations) - energy_kinetic(cons, equations)
end


# Calculate the error for the "lake-at-rest" test case where H = h_upper+h_lower+b should
# be a constant value over time
@inline function lake_at_rest_error(u, equations::ShallowWaterTwoLayerEquations1D)
  h_upper, _, h_lower, _, b = u
  return abs(equations.H0 - (h_upper + h_lower + b))
end

end # @muladd
