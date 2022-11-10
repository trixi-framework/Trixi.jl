# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin


@doc raw"""
    TwoLayerShallowWaterEquations1D(gravity, H0, rho1, rho2)

Two-Layer Shallow water equations (SWE) in one space dimension. The equations are given by
```math
\begin{aligned}
  \frac{\partial h_1}{\partial t} + \frac{\partial}{\partial x}(h_1 v_1) &= 0 \\
  \frac{\partial h_2}{\partial t} + \frac{\partial}{\partial x}(h_2 v_2) &= 0 \\
    \frac{\partial}{\partial t}(h1 v1) + \frac{\partial}{\partial x}\left(h1 v1^2 + \frac{g}{2}h1^2 
    \right) &= - g h_1 \frac{\partial}{\partial x}\left(b + \frac{\rho_2}{\rho_1}h_2 \right) \\
    \frac{\partial}{\partial t}(h2 v2) + \frac{\partial}{\partial x}\left(h2 v2^2 + \frac{g}{2}h2^2 
    \right)&= - g h_2 \frac{\partial}{\partial x}\left(b + h_1 \right)
\end{aligned}
```
The unknown quantities of the SWE are the water heights of the lower layer ``h_1`` and the upper 
layer ``h_2`` and the respecitve velocities ``v_1`` and ``v_2``. The gravitational constant is 
denoted by `g`, the layer densitites by ``\rho_1``and ``\rho_2`` and the (possibly) variable bottom 
topography function ``b(x)``. Conservative variable water height ``h_1`` is measured from the bottom
topography ``b`` and ``h_2`` relative to ``h_1``, therefore one also defines the total water heights
as ``H1 = h1 + b`` and ``H2 = h2 + h1 + b``.

The additional quantity ``H_0`` is also available to store a reference value for the total water
height that is useful to set initial conditions or test the "lake-at-rest" well-balancedness.

The bottom topography function ``b(x)`` is set inside the initial condition routine
for a particular problem setup. To test the conservative form of the SWE one can set the bottom 
topography variable `b` to zero.

In addition to the unknowns, Trixi currently stores the bottom topography values at the approximation
points despite being fixed in time. This is done for convenience of computing the bottom topography
gradients on the fly during the approximation as well as computing auxiliary quantities like the 
total water height ``H`` or the entropy variables.
This affects the implementation and use of these equations in various ways:
* The flux values corresponding to the bottom topography must be zero.
* The bottom topography values must be included when defining initial conditions, boundary 
  conditions or source terms.
* [`AnalysisCallback`](@ref) analyzes this variable.
* Trixi's visualization tools will visualize the bottom topography by default.

References for the SWE are many but a good introduction is available in Chapter 13 of the book:
- Randall J. LeVeque (2002)
  Finite Volume Methods for Hyperbolic Problems
  [DOI: 10.1017/CBO9780511791253](https://doi.org/10.1017/CBO9780511791253)
"""
struct TwoLayerShallowWaterEquations1D{RealT<:Real} <: AbstractShallowWaterEquations{1, 5}
  gravity::RealT # gravitational constant
  H0::RealT      # constant "lake-at-rest" total water height
  rho1::RealT    # lower layer density
  rho2::RealT    # upper layer density
end

# Allow for flexibility to set the gravitational constant within an elixir depending on the
# application where `gravity_constant=1.0` or `gravity_constant=9.81` are common values.
# The reference total water height H0 defaults to 0.0 but is used for the "lake-at-rest"
# well-balancedness test cases. Densities must be specificed such that rho_1 < rho_2.
function TwoLayerShallowWaterEquations1D(; gravity_constant, H0=0.0, rho1, rho2)
  TwoLayerShallowWaterEquations1D(gravity_constant, H0, rho1, rho2)
end


have_nonconservative_terms(::TwoLayerShallowWaterEquations1D) = Val(true)
varnames(::typeof(cons2cons), ::TwoLayerShallowWaterEquations1D) = ("h1", "h2", "h1_v1", "h2_v2", "b")
# Note, we use the total water height, H2 = h1 + h2 + b, and first layer total heigth H1 = h1 + b as
# the first primitive variable for easier visualization and setting initial conditions
varnames(::typeof(cons2prim), ::TwoLayerShallowWaterEquations1D) = ("H1", "H2", "v1", "v2", "b")


# Set initial conditions at physical location `x` for time `t`
"""
    initial_condition_convergence_test(x, t, equations::TwoLayerShallowWaterEquations1D)

A smooth initial condition used for convergence tests in combination with
[`source_terms_convergence_test`](@ref)
(and [`BoundaryConditionDirichlet(initial_condition_convergence_test)`](@ref) in non-periodic domains).
"""
function initial_condition_convergence_test(x, t, equations::TwoLayerShallowWaterEquations1D)
  # some constants are chosen such that the function is periodic on the domain [0,sqrt(2)]

  ω = 2 * pi * sqrt(2.0)

  v1 = 1.0
  v2 = 0.9 
  H1 = 2.0 + 0.1sin(ω*x[1]+t)
  H2 = 4.0 + 0.1cos(ω*x[1]+t) 
  b  = 1.0 + 0.1cos(2*ω*x[1])
  return prim2cons(SVector(H1, H2, v1, v2, b), equations)
end

"""
    source_terms_convergence_test(u, x, t, equations::TwoLayerShallowWaterEquations1D)

Source terms used for convergence tests in combination with
[`initial_condition_convergence_test`](@ref)
(and [`BoundaryConditionDirichlet(initial_condition_convergence_test)`](@ref) in non-periodic domains).
"""

@inline function source_terms_convergence_test(u, x, t, equations::TwoLayerShallowWaterEquations1D)
  # Same settings as in `initial_condition_convergence_test`. Some derivative simplify because
  # this manufactured solution velocity is taken to be constant
  ω = 2* pi * sqrt(2.0)

  du1 = 0.1cos(t + ω*x[1]) + 0.1ω*cos(t + ω*x[1]) + 0.2ω*sin(2ω*x[1])
  du2 = (-0.1cos(t + ω*x[1]) - 0.1sin(t + ω*x[1]) - 0.09000000000000001ω*cos(t + ω*x[1]) -
        0.09000000000000001ω*sin(t + ω*x[1]))
  du3 = ((10.0 + sin(t + ω*x[1]) - cos(2ω*x[1]))*(-0.09000000000000001ω*cos(t + ω*x[1]) -
    0.09000000000000001ω*sin(t + ω*x[1]) - 0.2ω*sin(2ω*x[1])) + 0.1cos(t + ω*x[1]) + 0.1ω*cos(t +
    ω*x[1]) + 5.0(0.1ω*cos(t + ω*x[1]) + 0.2ω*sin(2ω*x[1]))*(2.0 + 0.2sin(t + ω*x[1]) -
    0.2cos(2ω*x[1])) + 0.2ω*sin(2ω*x[1]))
  du4 = (5.0(-0.1ω*cos(t + ω*x[1]) - 0.1ω*sin(t + ω*x[1]))*(4.0 + 0.2cos(t + ω*x[1]) - 0.2sin(t +
    ω*x[1])) + 0.1ω*(20.0 + cos(t + ω*x[1]) - sin(t + ω*x[1]))*cos(t + ω*x[1]) -
    0.09000000000000001cos(t + ω*x[1]) - 0.09000000000000001sin(t + ω*x[1]) -
    0.08100000000000002ω*cos(t + ω*x[1]) - 0.08100000000000002ω*sin(t + ω*x[1]))
  return SVector(du1, du2, du3, du4, 0.0)
end

"""
    boundary_condition_slip_wall(u_inner, orientation_or_normal, x, t, surface_flux_function,
                                  equations::TwoLayerShallowWaterEquations1D)

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
                                              x, t,
                                              surface_flux_function,
                                              equations::TwoLayerShallowWaterEquations1D)

  # create the "external" boundary solution state
  u_boundary = SVector(u_inner[1],
                       u_inner[2],
                       -u_inner[3],
                       -u_inner[4],
                       u_inner[5])
                    
  # calculate the boundary flux
  if iseven(direction) # u_inner is "left" of boundary, u_boundary is "right" of boundary
    flux = surface_flux_function(u_inner, u_boundary, orientation_or_normal, equations)
  else # u_boundary is "left" of boundary, u_inner is "right" of boundary
    flux = surface_flux_function(u_boundary, u_inner, orientation_or_normal, equations)
  end

  return flux
end

# Calculate 1D flux for a single point
# Note, the bottom topography has no flux
@inline function flux(u, orientation::Integer, equations::TwoLayerShallowWaterEquations1D)
  h1, h2, h1_v1, h2_v2, _ = u

  # Calculate velocities
  v1, v2 = velocity(u, equations)

  # Calculate pressure
  p1 = 0.5 * equations.gravity * h1^2
  p2 = 0.5 * equations.gravity * h2^2

  f1 = h1_v1
  f2 = h2_v2
  f3 = h1_v1 * v1 + p1
  f4 = h2_v2 * v2 + p2
  return SVector(f1, f2, f3, f4, zero(eltype(u)))
end

"""
    flux_nonconservative_wintermeyer_etal(u_ll, u_rr, orientation::Integer,
                                          equations::TwoLayerShallowWaterEquations1D)

Non-symmetric two-point volume flux discretizing the nonconservative (source) term
that contains the gradient of the bottom topography [`TwoLayerShallowWaterEquations1D`](@ref).

Further details are available in the paper:
- Niklas Wintermeyer, Andrew R. Winters, Gregor J. Gassner and David A. Kopriva (2017)
  An entropy stable nodal discontinuous Galerkin method for the two dimensional
  shallow water equations on unstructured curvilinear meshes with discontinuous bathymetry
  [DOI: 10.1016/j.jcp.2017.03.036](https://doi.org/10.1016/j.jcp.2017.03.036)
"""
@inline function flux_nonconservative_wintermeyer_etal(u_ll, u_rr, orientation::Integer,
                                                         equations::TwoLayerShallowWaterEquations1D)
  # Pull the necessary left and right state information
  h1_ll, h2_ll = waterheight(u_ll, equations)
  h1_rr, h2_rr = waterheight(u_rr, equations)
  b_rr = u_rr[5]

  z = zero(eltype(u_ll))

  # Bottom gradient nonconservative term: (0, 0, g*h1*(b+rh2)_x , g*h2*(b+h1)_x, 0)
  f = SVector(z, z, 
              equations.gravity * h1_ll * (b_rr + equations.rho2/equations.rho1 * h2_rr),
              equations.gravity * h2_ll * (b_rr + h1_rr),
              z)
  return f
end

"""
    flux_nonconservative_fjordholm_etal(u_ll, u_rr, orientation::Integer,
                                          equations::TwoLayerShallowWaterEquations1D)

Non-symmetric two-point surface flux discretizing the nonconservative (source) term that contains 
the gradients of the bottom topography and the layer heights [`TwoLayerShallowWaterEquations1D`](@ref).

Further details are available in the paper:
- Ulrik Skre Fjordholm (2012)
  Energy conservative and stable schemes for the two-layer shallow water equations.
  (https://doi.org/10.1142/9789814417099_0039)
"""
@inline function flux_nonconservative_fjordholm_etal(u_ll, u_rr, orientation::Integer,
                                                         equations::TwoLayerShallowWaterEquations1D)
  # Pull the necessary left and right state information
  h1_ll, h2_ll, _, _, b_ll = u_ll
  h1_rr, h2_rr, _, _, b_rr = u_rr

  # Create average and jump values
  h1_average = 0.5 * (h1_ll + h1_rr)
  h2_average = 0.5 * (h2_ll + h2_rr)
  h1_jump    = h1_rr - h1_ll
  h2_jump    = h2_rr - h2_ll
  b_jump = b_rr - b_ll

  # Assign variables for constants for better readability
  g = equations.gravity
  r = equations.rho2 / equations.rho1

  z = zero(eltype(u_ll))

  # Bottom gradient nonconservative term: (0, 0, g*h1*(b+rh2)_x , g*h2*(b+h1)_x, 0)
  f = SVector(z,z,
                g*h1_ll*(b_ll + r*h2_ll) + g*h1_average*(b_jump + r*h2_jump),
                g*h2_ll*(b_ll + h1_ll)   + g*h2_average*(b_jump + h1_jump), 
                z)
  return f
end

"""
    flux_fjordholm_etal(u_ll, u_rr, orientation,
                        equations::TwoLayerShallowWaterEquations1D)

Total energy conservative (mathematical entropy for shallow water equations). When the bottom 
topography is nonzero this should only be used as a surface flux otherwise the scheme will not be 
well-balanced. For well-balancedness in the volume flux use [`flux_wintermeyer_etal`](@ref).

Details are available in Eq. (4.1) in the paper:
- Ulrik S. Fjordholm, Siddhartha Mishr and Eitan Tadmor (2011)
  Well-balanced and energy stable schemes for the shallow water equations with discontinuous 
  topography [DOI: 10.1016/j.jcp.2011.03.042](https://doi.org/10.1016/j.jcp.2011.03.042)
and the application to two layers is shown in the paper:
- Ulrik Skre Fjordholm (2012)
  Energy conservative and stable schemes for the two-layer shallow water equations.
  (https://doi.org/10.1142/9789814417099_0039)
"""
@inline function flux_fjordholm_etal(u_ll, u_rr, orientation::Integer, 
                                                         equations::TwoLayerShallowWaterEquations1D)
  # Unpack left and right state
  h1_ll, h2_ll = waterheight(u_ll, equations)
  v1_ll, v2_ll = velocity(u_ll, equations)
  h1_rr, h2_rr = waterheight(u_rr, equations)
  v1_rr, v2_rr = velocity(u_rr, equations)

  # Average each factor of products in flux
  h1_avg = 0.5 * (h1_ll + h1_rr )
  h2_avg = 0.5 * (h2_ll + h2_rr )
  v1_avg = 0.5 * (v1_ll + v1_rr )
  v2_avg = 0.5 * (v2_ll + v2_rr )
  p1_avg = 0.25* equations.gravity * (h1_ll^2 + h1_rr^2)
  p2_avg = 0.25* equations.gravity * (h2_ll^2 + h2_rr^2)

  # Calculate fluxes depending on orientation
  f1 = h1_avg * v1_avg
  f2 = h2_avg * v2_avg
  f3 = f1 * v1_avg + p1_avg
  f4 = f2 * v2_avg + p2_avg

  return SVector(f1, f2, f3, f4, zero(eltype(u_ll)))
end


"""
    flux_wintermeyer_etal(u_ll, u_rr, orientation,
                          equations::TwoLayerShallowWaterEquations1D)

Total energy conservative (mathematical entropy for shallow water equations) split form.
When the bottom topography is nonzero this scheme will be well-balanced when used as a `volume_flux`.
The `surface_flux` should still use, e.g., [`flux_fjordholm_etal`](@ref).
-> Adapted for TwoLayerShallowWater

Further details are available in Theorem 1 of the paper:
- Niklas Wintermeyer, Andrew R. Winters, Gregor J. Gassner and David A. Kopriva (2017)
  An entropy stable nodal discontinuous Galerkin method for the two dimensional
  shallow water equations on unstructured curvilinear meshes with discontinuous bathymetry
  [DOI: 10.1016/j.jcp.2017.03.036](https://doi.org/10.1016/j.jcp.2017.03.036)
"""
@inline function flux_wintermeyer_etal(u_ll, u_rr, orientation::Integer, 
                                                         equations::TwoLayerShallowWaterEquations1D)
  # Unpack left and right state
  h1_ll, h2_ll, h1_v1_ll, h2_v2_ll, _ = u_ll
  h1_rr, h2_rr, h1_v1_rr, h2_v2_rr, _ = u_rr

  # Get the velocities on either side
  v1_ll, v2_ll = velocity(u_ll, equations)
  v1_rr, v2_rr = velocity(u_rr, equations)

  # Average each factor of products in flux
  v1_avg = 0.5 * (v1_ll + v1_rr)
  v2_avg = 0.5 * (v2_ll + v2_rr)
  p1_avg = 0.5 * equations.gravity * h1_ll * h1_rr
  p2_avg = 0.5 * equations.gravity * h2_ll * h2_rr

  # Calculate fluxes depending on orientation
  f1 = 0.5 * (h1_v1_ll + h1_v1_rr)
  f2 = 0.5 * (h2_v2_ll + h2_v2_rr)
  f3 = f1 * v1_avg + p1_avg
  f4 = f2 * v2_avg + p2_avg

  return SVector(f1, f2, f3, f4, zero(eltype(u_ll)))
end


# Calculate maximum wave speed for local Lax-Friedrichs-type dissipation as the
# maximum velocity magnitude plus the maximum speed of sound as approximated by fjordholm
@inline function max_abs_speed_naive(u_ll, u_rr, orientation::Integer, 
  equations::TwoLayerShallowWaterEquations1D)
# Get the averaged velocity
Um_ll = (u_ll[3] + u_ll[4]) / (u_ll[1] + u_ll[2])
Um_rr = (u_rr[3] + u_rr[4]) / (u_rr[1] + u_rr[2])

# Calculate the wave celerity on the left and right
h1_ll, h2_ll = waterheight(u_ll, equations)
h1_rr, h2_rr = waterheight(u_rr, equations)
c_ll = sqrt(equations.gravity * (h1_ll + h2_ll) )
c_rr = sqrt(equations.gravity * (h1_rr + h2_rr))
return (max(abs(Um_ll) + c_ll, abs(Um_rr) + c_rr))
end


# Specialized `DissipationLocalLaxFriedrichs` to avoid spurious dissipation in the bottom topography
@inline function (dissipation::DissipationLocalLaxFriedrichs)(u_ll, u_rr, 
                        orientation_or_normal_direction, equations::TwoLayerShallowWaterEquations1D)
  λ = dissipation.max_abs_speed(u_ll, u_rr, orientation_or_normal_direction, equations)
  diss = -0.5 * λ * (u_rr - u_ll)
  return SVector(diss[1], diss[2], diss[3], diss[4], zero(eltype(u_ll)))
end


# Absolute speed according to Fjordholm
@inline function max_abs_speeds(u, equations::TwoLayerShallowWaterEquations1D)

  v = (u[3] + u[4]) / (u[1] + u[2])
  h1, h2 = waterheight(u, equations)

  c = sqrt(equations.gravity*(h1 + h2)) 
  return (abs(v) + c)
end


# Helper function to extract the velocity vector from the conservative variables
@inline function velocity(u, equations::TwoLayerShallowWaterEquations1D)
  h1, h2, h1_v1, h2_v2, _ = u

  v1 = h1_v1 / h1
  v2 = h2_v2 / h2

  return v1, v2
end


# Convert conservative variables to primitive
@inline function cons2prim(u, equations::TwoLayerShallowWaterEquations1D)
  h1, h2,_, _, b = u

  H1 = h1 + b
  H2 = h2 + H1
  v1,v2 = velocity(u, equations)
  return SVector(H1, H2, v1, v2, b)
end


# Convert conservative variables to entropy
# Note, only the first four are the entropy variables, the fifth entry still
# just carries the bottom topography values for convenience
@inline function cons2entropy(u, equations::TwoLayerShallowWaterEquations1D)
  h1, h2, h1_v1, h2_v2, b = u
  ρ1 = equations.rho1
  ρ2 = equations.rho2

  v1, v2 = velocity(u, equations)

  w1 = ρ1 * (equations.gravity * (h1 + (ρ2/ρ1)*h2 + b) - 0.5 * v1^2)
  w2 = ρ2 * (equations.gravity * (h1 +         h2 + b) - 0.5 * v2^2)
  w3 = ρ1 * v1
  w4 = ρ2 * v2

  return SVector(w1, w2, w3, w4, b)
end


# Convert primitive to conservative variables
@inline function prim2cons(prim, equations::TwoLayerShallowWaterEquations1D)
  H1, H2, v1, v2, b = prim

  h1 = H1 - b
  h2 = H2 - h1 - b
  h1_v1 = h1 * v1
  h2_v2 = h2 * v2
  return SVector(h1, h2, h1_v1, h2_v2, b)
end


@inline function waterheight(u, equations::TwoLayerShallowWaterEquations1D)
  return u[1],u[2]
end


# Entropy function for the shallow water equations is the total energy
@inline entropy(cons, equations::TwoLayerShallowWaterEquations1D) = energy_total(cons, equations)


# Calculate total energy for a conservative state `cons`
@inline function energy_total(cons, equations::TwoLayerShallowWaterEquations1D)
  h1, h2, h1_v1, h2_v2, b = cons
  g = equations.gravity
  ρ1= equations.rho1
  ρ2= equations.rho2

  e = 0.5*ρ1 * (h1_v1^2/h1 + g*h1^2) + 0.5*ρ2 * (h2_v2^2/h2 + g*h2^2) + g*ρ1*h1*b + g*ρ2*h2*(h1 + b)
  return e
end


# Calculate kinetic energy for a conservative state `cons`
@inline function energy_kinetic(u, equations::TwoLayerShallowWaterEquations1D)
  h1, h2, h1_v1, h2_v2, _ = u
  ρ1 = equations.rho1
  ρ2 = equations.rho2

  0.5*ρ1 * h1_v1^2/h1 + 0.5*ρ2*h2_v2^2/h2
  return 
end


# Calculate potential energy for a conservative state `cons`
@inline function energy_internal(cons, equations::TwoLayerShallowWaterEquations1D)
  return energy_total(cons, equations) - energy_kinetic(cons, equations)
end


# Calculate the error for the "lake-at-rest" test case where H = h1+h2+b should
# be a constant value over time
@inline function lake_at_rest_error(u, equations::TwoLayerShallowWaterEquations1D)
  h1, h2, _, _, b = u
  return abs(equations.H0 - (h1 + h2 + b))
end

end # @muladd
