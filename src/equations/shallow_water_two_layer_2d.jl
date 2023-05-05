# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin

@doc raw"""
    ShallowWaterTwoLayerEquations2D(gravity, H0, rho_upper, rho_lower)

Two-Layer Shallow water equations (2LSWE) in two space dimension. The equations are given by
```math
\begin{alignat*}{8}
&\frac{\partial}{\partial t}h_{upper}        
&&+ \frac{\partial}{\partial x}\left(h_{upper} v_{1,upper}\right)
&&+ \frac{\partial}{\partial y}\left(h_{upper} v_{2,upper}\right)  \quad 
&&= \quad 0 \\
&\frac{\partial}{\partial t}\left(h_{upper} v_{1,upper}\right)  
&&+ \frac{\partial}{\partial x}\left(h_{upper} v_{1,upper}^2 + \frac{gh_{upper}^2}{2}\right) 
&&+ \frac{\partial}{\partial y}\left(h_{upper} v_{1,upper} v_{2,upper}\right) \quad 
&&= -gh_{upper}\frac{\partial}{\partial x}\left(b+h_{lower}\right) \\
&\frac{\partial}{\partial t}\left(h_{upper} v_{2,upper}\right) 
&&+ \frac{\partial}{\partial x}\left(h_{upper} v_{1,upper} v_{2,upper}\right) 
&&+ \frac{\partial}{\partial y}\left(h_{upper} v_{2,upper}^2 + \frac{gh_{upper}^2}{2}\right) 
&&= -gh_{upper}\frac{\partial}{\partial y}\left(b+h_{lower}\right)\\
&\frac{\partial}{\partial t}h_{lower}  
&&+ \frac{\partial}{\partial x}\left(h_{lower} v_{1,lower}\right) 
&&+ \frac{\partial}{\partial y}\left(h_{lower} v_{2,lower}\right) 
&&= \quad 0 \\
&\frac{\partial}{\partial t}\left(h_{lower} v_{1,lower}\right) 
&&+ \frac{\partial}{\partial x}\left(h_{lower} v_{1,lower}^2 + \frac{gh_{lower}^2}{2}\right) 
&&+ \frac{\partial}{\partial y}\left(h_{lower} v_{1,lower} v_{2,lower}\right) 
&&= -gh_{lower}\frac{\partial}{\partial x}\left(b+\frac{\rho_{upper}}{\rho_{lower}} h_{upper}\right)\\
&\frac{\partial}{\partial t}\left(h_{lower} v_{2,lower}\right)  
&&+ \frac{\partial}{\partial x}\left(h_{lower} v_{1,lower} v_{2,lower}\right) 
&&+ \frac{\partial}{\partial y}\left(h_{lower} v_{2,lower}^2 + \frac{gh_{lower}^2}{2}\right) 
&&= -gh_{lower}\frac{\partial}{\partial y}\left(b+\frac{\rho_{upper}}{\rho_{lower}} h_{upper}\right)
\end{alignat*}
```
The unknown quantities of the 2LSWE are the water heights of the lower layer ``h_{lower}`` and the 
upper 
layer ``h_{upper}`` and the respective velocities in x-direction ``v_{1,lower}`` and ``v_{1,upper}`` and in y-direction
``v_{2,lower}`` and ``v_{2,upper}``. The gravitational constant is denoted by `g`, the layer densitites by 
``\rho_{upper}``and ``\rho_{lower}`` and the (possibly) variable bottom topography function by ``b(x)``. 
Conservative variable water height ``h_{lower}`` is measured from the bottom topography ``b`` and ``h_{upper}`` 
relative to ``h_{lower}``, therefore one also defines the total water heights as ``H_{lower} = h_{lower} + b`` and 
``H_{upper} = h_{upper} + h_{lower} + b``.

The densities must be chosen such that ``\rho_{upper} < \rho_{lower}``, to make sure that the heavier fluid 
``\rho_{lower}`` is in the bottom layer and the lighter fluid ``\rho_{upper}`` in the upper layer.

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
struct ShallowWaterTwoLayerEquations2D{RealT<:Real} <: AbstractShallowWaterEquations{2, 7}
  gravity::RealT   # gravitational constant
  H0::RealT        # constant "lake-at-rest" total water height
  rho_upper::RealT # lower layer density
  rho_lower::RealT # upper layer density
  r::RealT         # ratio of rho_upper / rho_lower
end

# Allow for flexibility to set the gravitational constant within an elixir depending on the
# application where `gravity_constant=1.0` or `gravity_constant=9.81` are common values.
# The reference total water height H0 defaults to 0.0 but is used for the "lake-at-rest"
# well-balancedness test cases. Densities must be specified such that rho_upper < rho_lower.
function ShallowWaterTwoLayerEquations2D(; gravity_constant, H0=zero(gravity_constant), rho_upper, rho_lower)
  # Assign density ratio if rho_upper <= rho_lower
  if rho_upper > rho_lower
    error("Invalid input: Densities must be chosen such that rho_upper <= rho_lower")
  else
    r = rho_upper / rho_lower
  end
  ShallowWaterTwoLayerEquations2D(gravity_constant, H0, rho_upper, rho_lower, r)
end


have_nonconservative_terms(::ShallowWaterTwoLayerEquations2D) = True()
varnames(::typeof(cons2cons), ::ShallowWaterTwoLayerEquations2D) = (
    "h_upper", "h_v1_upper", "h_v2_upper", "h_lower", "h_v1_lower", "h_v2_lower", "b")                                                             
# Note, we use the total water height, H_upper = h_upper + h_lower + b, and first layer total height
# H_lower = h_lower + b as the first primitive variable for easier visualization and setting initial
# conditions
varnames(::typeof(cons2prim), ::ShallowWaterTwoLayerEquations2D) = (
    "H_upper", "v1_upper", "v2_upper", "H_lower", "v1_lower", "v2_lower", "b")


# Set initial conditions at physical location `x` for time `t`
"""
    initial_condition_convergence_test(x, t, equations::ShallowWaterTwoLayerEquations2D)

A smooth initial condition used for convergence tests in combination with
[`source_terms_convergence_test`](@ref). Constants must be set to ``rho_{upper} = 0.9``, 
``rho_{lower} = 1.0``, ``g = 10.0``.
"""
function initial_condition_convergence_test(x, t, equations::ShallowWaterTwoLayerEquations2D)
  # some constants are chosen such that the function is periodic on the domain [0,sqrt(2)]^2]
  ω = 2.0 * pi * sqrt(2.0)

  H_lower  = 2.0 + 0.1 * sin(ω * x[1] + t) * cos(ω * x[2] + t)
  H_upper  = 4.0 + 0.1 * cos(ω * x[1] + t) * sin(ω * x[2] + t)
  v1_lower = 1.0
  v1_upper = 0.9
  v2_lower = 0.9
  v2_upper = 1.0
  b        = 1.0 + 0.1 * cos(0.5 * ω * x[1]) * sin(0.5 * ω * x[2])

  return prim2cons(SVector(H_upper, v1_upper, v2_upper, H_lower, v1_lower, v2_lower, b), equations)
end


"""
    source_terms_convergence_test(u, x, t, equations::ShallowWaterTwoLayerEquations2D)

Source terms used for convergence tests in combination with
[`initial_condition_convergence_test`](@ref).
"""
@inline function source_terms_convergence_test(u, x, t, equations::ShallowWaterTwoLayerEquations2D)
  # Same settings as in `initial_condition_convergence_test`. 
  # some constants are chosen such that the function is periodic on the domain [0,sqrt(2)]^2]
  ω = 2.0 * pi * sqrt(2.0)

  # Source terms obtained with SymPy
  du1 = 0.01*ω*cos(t + ω*x[1])*cos(t + ω*x[2]) + 0.01*ω*sin(t + ω*x[1])*sin(t + ω*x[2])
  du2 = (5.0 * (-0.1*ω*cos(t + ω*x[1])*cos(t + ω*x[2]) - 0.1*ω*sin(t + ω*x[1])*sin(t +
         ω*x[2])) * (4.0 + 0.2cos(t + ω*x[1])*sin(t + ω*x[2]) - 0.2*sin(t + ω*x[1])*cos(t +
         ω*x[2])) + 0.009*ω*cos(t + ω*x[1])*cos(t + ω*x[2]) + 0.009*ω*sin(t + ω*x[1])*sin(t +
         ω*x[2]) + 0.1*ω*(20.0 + cos(t + ω*x[1])*sin(t + ω*x[2]) - sin(t + ω*x[1])*cos(t +
         ω*x[2])) * cos(t + ω*x[1])*cos(t + ω*x[2]))
  du3 = (5.0 * (0.1*ω*cos(t + ω*x[1])*cos(t + ω*x[2]) + 0.1*ω*sin(t + ω*x[1])*sin(t +
         ω*x[2])) * (4.0 + 0.2*cos(t + ω*x[1])*sin(t + ω*x[2]) - 0.2*sin(t + ω*x[1])*cos(t +
         ω*x[2])) + 0.01ω*cos(t + ω*x[1])*cos(t + ω*x[2]) + 0.01*ω*sin(t + ω*x[1])*sin(t + ω*x[2]) +
        -0.1*ω*(20.0 + cos(t + ω*x[1])*sin(t + ω*x[2]) - sin(t + ω*x[1])*cos(t + ω*x[2]))*sin(t +
         ω*x[1])*sin(t + ω*x[2]))
  du4 = (0.1*cos(t + ω*x[1])*cos(t + ω*x[2]) + 0.1*ω*cos(t + ω*x[1])*cos(t + ω*x[2]) +
        0.05*ω*sin(0.5*ω*x[1])*sin(0.5*ω*x[2]) - 0.1*sin(t + ω*x[1])*sin(t + ω*x[2]) +
        -0.045*ω*cos(0.5*ω*x[1])*cos(0.5*ω*x[2]) - 0.09*ω*sin(t + ω*x[1])*sin(t + ω*x[2]))
  du5 = ((10.0 + sin(t + ω*x[1])*cos(t + ω*x[2]) - cos(0.5*ω*x[1])*sin(0.5*ω*x[2]))*(-0.09*ω*cos(t +
         ω*x[1])*cos(t + ω*x[2]) - 0.09*ω*sin(t + ω*x[1])*sin(t + ω*x[2]) +
        -0.05*ω*sin(0.5*ω*x[1])*sin(0.5*ω*x[2])) + 5.0 * (0.1*ω*cos(t + ω*x[1])*cos(t + ω*x[2]) +
         0.05*ω*sin(0.5*ω*x[1])*sin(0.5*ω*x[2])) * (2.0 + 0.2*sin(t + ω*x[1])*cos(t + ω*x[2]) +
        -0.2*cos(0.5*ω*x[1])*sin(0.5*ω*x[2])) + 0.1*cos(t + ω*x[1])*cos(t + ω*x[2]) + 0.1*ω*cos(t +
         ω*x[1])*cos(t + ω*x[2]) + 0.05*ω*sin(0.5*ω*x[1])*sin(0.5*ω*x[2]) - 0.1*sin(t +
         ω*x[1])*sin(t + ω*x[2]) - 0.045*ω*cos(0.5*ω*x[1])*cos(0.5*ω*x[2]) - 0.09*ω*sin(t +
         ω*x[1])*sin(t + ω*x[2]))
  du6 = ((10.0 + sin(t + ω*x[1])*cos(t + ω*x[2]) +
          -cos(0.5*ω*x[1])*sin(0.5*ω*x[2])) * (0.05*ω*cos(0.5*ω*x[1])*cos(0.5*ω*x[2]) +
          0.09*ω*cos(t + ω*x[1])*cos(t + ω*x[2]) + 0.09*ω*sin(t + ω*x[1])*sin(t + ω*x[2])) +
          5.0 * (-0.05*ω*cos(0.5*ω*x[1])*cos(0.5*ω*x[2]) - 0.1*ω*sin(t + ω*x[1])*sin(t + 
          ω*x[2])) * (2.0 + 0.2*sin(t + ω*x[1])*cos(t + ω*x[2]) +
         -0.2*cos(0.5*ω*x[1])*sin(0.5*ω*x[2])) + 0.09cos(t + ω*x[1])*cos(t + ω*x[2]) +
          0.09*ω*cos(t + ω*x[1])*cos(t + ω*x[2]) + 0.045*ω*sin(0.5*ω*x[1])*sin(0.5*ω*x[2]) +
         -0.09*sin(t + ω*x[1])*sin(t + ω*x[2]) - 0.0405*ω*cos(0.5*ω*x[1])*cos(0.5*ω*x[2]) +
         -0.081*ω*sin(t + ω*x[1])*sin(t + ω*x[2]))

  return SVector(du1, du2, du3, du4, du5, du6, zero(eltype(u)))
end


"""
    boundary_condition_slip_wall(u_inner, normal_direction, x, t, surface_flux_function,
                                 equations::ShallowWaterTwoLayerEquations2D)

Create a boundary state by reflecting the normal velocity component and keep
the tangential velocity component unchanged. The boundary water height is taken from
the internal value.

For details see Section 9.2.5 of the book:
- Eleuterio F. Toro (2001)
  Shock-Capturing Methods for Free-Surface Shallow Flows
  1st edition
  ISBN 0471987662
"""
@inline function boundary_condition_slip_wall(u_inner, normal_direction::AbstractVector,
                                              x, t, surface_flux_function,
                                              equations::ShallowWaterTwoLayerEquations2D)
  # normalize the outward pointing direction
  normal = normal_direction / norm(normal_direction)

  # compute the normal velocity
  v_normal_upper = normal[1] * u_inner[2] + normal[2] * u_inner[3]
  v_normal_lower = normal[1] * u_inner[5] + normal[2] * u_inner[6]

  # create the "external" boundary solution state
  u_boundary = SVector(u_inner[1],
                       u_inner[2] - 2.0 * v_normal_upper * normal[1],
                       u_inner[3] - 2.0 * v_normal_upper * normal[2],
                       u_inner[4],
                       u_inner[5] - 2.0 * v_normal_lower * normal[1],
                       u_inner[6] - 2.0 * v_normal_lower * normal[2],
                       u_inner[7])

  # calculate the boundary flux
  flux = surface_flux_function(u_inner, u_boundary, normal_direction, equations)
  return flux
end


# Calculate 1D flux for a single point
# Note, the bottom topography has no flux
@inline function flux(u, orientation::Integer, equations::ShallowWaterTwoLayerEquations2D)
  h_upper, h_v1_upper, h_v2_upper, h_lower, h_v1_lower, h_v2_lower, _ = u

  # Calculate velocities
  v1_upper, v2_upper, v1_lower, v2_lower = velocity(u, equations)

  # Calculate pressure
  p1 = 0.5 * equations.gravity * h_upper^2
  p2 = 0.5 * equations.gravity * h_lower^2

  # Calculate fluxes depending on orientation
  if orientation == 1
    f1 = h_v1_upper
    f2 = h_v1_upper * v1_upper + p1
    f3 = h_v1_upper * v2_upper
    f4 = h_v1_lower
    f5 = h_v1_lower * v1_lower + p2
    f6 = h_v1_lower * v2_lower
  else
    f1 = h_v2_upper
    f2 = h_v2_upper * v1_upper
    f3 = h_v2_upper * v2_upper + p1
    f4 = h_v2_lower
    f5 = h_v2_lower * v1_lower
    f6 = h_v2_lower * v2_lower + p2
  end
  return SVector(f1, f2, f3, f4, f5 , f6, zero(eltype(u)))
end

# Calculate 1D flux for a single point in the normal direction
# Note, this directional vector is not normalized and the bottom topography has no flux
@inline function flux(u, normal_direction::AbstractVector, 
                      equations::ShallowWaterTwoLayerEquations2D)
  h_upper, h_lower = waterheight(u, equations)
  v1_upper, v2_upper, v1_lower, v2_lower = velocity(u, equations)

  v_normal_upper   = v1_upper * normal_direction[1] + v2_upper * normal_direction[2]
  v_normal_lower   = v1_lower * normal_direction[1] + v2_lower * normal_direction[2]
  h_v_upper_normal = h_upper * v_normal_upper
  h_v_lower_normal = h_lower * v_normal_lower

  p1 = 0.5 * equations.gravity * h_upper^2
  p2 = 0.5 * equations.gravity * h_lower^2

  f1 = h_v_upper_normal
  f2 = h_v_upper_normal * v1_upper + p1 * normal_direction[1]
  f3 = h_v_upper_normal * v2_upper + p1 * normal_direction[2]
  f4 = h_v_lower_normal
  f5 = h_v_lower_normal * v1_lower + p2 * normal_direction[1]
  f6 = h_v_lower_normal * v2_lower + p2 * normal_direction[2]

  return SVector(f1, f2, f3, f4, f5, f6, zero(eltype(u)))
end


"""
    flux_nonconservative_wintermeyer_etal(u_ll, u_rr, orientation::Integer,
                                          equations::ShallowWaterTwoLayerEquations2D)

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
                                                       equations::ShallowWaterTwoLayerEquations2D)
  # Pull the necessary left and right state information
  h_upper_ll, h_lower_ll = waterheight(u_ll, equations)
  h_upper_rr, h_lower_rr = waterheight(u_rr, equations)
  b_rr = u_rr[7]

  z = zero(eltype(u_ll))

  # Bottom gradient nonconservative term: (0, g*h_upper*(b + h_lower)_x, g*h_upper*(b + h_lower)_y ,
  #                                        0, g*h_lower*(b + r*h_upper)_x, 
  #                                        g*h_lower*(b + r*h_upper)_y, 0)
  if orientation == 1
    f = SVector(z,
    equations.gravity * h_upper_ll * (b_rr + h_lower_rr),
    z,z,
    equations.gravity * h_lower_ll * (b_rr + equations.r * h_upper_rr),
    z,z)
  else # orientation == 2
    f = SVector(z, z,
    equations.gravity * h_upper_ll * (b_rr + h_lower_rr),
    z,z,
    equations.gravity * h_lower_ll * (b_rr + equations.r * h_upper_rr),
    z)
  end

  return f
end

@inline function flux_nonconservative_wintermeyer_etal(u_ll, u_rr,
                                                       normal_direction_ll::AbstractVector,
                                                       normal_direction_average::AbstractVector,
                                                       equations::ShallowWaterTwoLayerEquations2D)
  # Pull the necessary left and right state information
  h_upper_ll, h_lower_ll = waterheight(u_ll, equations)
  h_upper_rr, h_lower_rr = waterheight(u_rr, equations)
  b_rr = u_rr[7]

  # Note this routine only uses the `normal_direction_average` and the average of the
  # bottom topography to get a quadratic split form DG gradient on curved elements
  return SVector(zero(eltype(u_ll)),
                 normal_direction_average[1] * equations.gravity * h_upper_ll * (b_rr + h_lower_rr),
                 normal_direction_average[2] * equations.gravity * h_upper_ll * (b_rr + h_lower_rr),
                 zero(eltype(u_ll)),
                 normal_direction_average[1] * equations.gravity * h_lower_ll * (b_rr + 
                    equations.r * h_upper_rr),
                 normal_direction_average[2] * equations.gravity * h_lower_ll * (b_rr +
                    equations.r * h_upper_rr),
                 zero(eltype(u_ll)))
  end


"""
    flux_nonconservative_fjordholm_etal(u_ll, u_rr, orientation::Integer,
                                        equations::ShallowWaterTwoLayerEquations2D)

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
                                                     equations::ShallowWaterTwoLayerEquations2D)
  # Pull the necessary left and right state information
  h_upper_ll, h_v1_upper_ll, h_v2_upper_ll, h_lower_ll, h_v1_lower_ll, h_v2_lower_ll, b_ll = u_ll
  h_upper_rr, h_v1_upper_rr, h_v2_upper_rr, h_lower_rr, h_v1_lower_rr, h_v2_lower_rr, b_rr = u_rr

  # Create average and jump values
  h_upper_average = 0.5 * (h_upper_ll + h_upper_rr)
  h_lower_average = 0.5 * (h_lower_ll + h_lower_rr)
  h_upper_jump    = h_upper_rr - h_upper_ll
  h_lower_jump    = h_lower_rr - h_lower_ll
  b_jump     = b_rr  - b_ll

  # Assign variables for constants for better readability
  g = equations.gravity

  # Bottom gradient nonconservative term: (0, g*h_upper*(b+h_lower)_x, g*h_upper*(b+h_lower)_y, 0,
  #                                        g*h_lower*(b+r*h_upper)_x, g*h_lower*(b+r*h_upper)_x, 0)

  # Includes two parts:
  #   (i)  Diagonal (consistent) term from the volume flux that uses `b_ll` to avoid
  #        cross-averaging across a discontinuous bottom topography
  #   (ii) True surface part that uses `h_average` and `b_jump` to handle discontinuous bathymetry
  z = zero(eltype(u_ll))
  if orientation == 1 
    f = SVector(
      z,
      g * h_upper_ll * (b_ll +   h_lower_ll) + g * h_upper_average * (b_jump +   h_lower_jump),
      z,z,
      g * h_lower_ll * (b_ll + equations.r * h_upper_ll) + g * h_lower_average * (b_jump +
          equations.r * h_upper_jump),
      z,z)
  else # orientation == 2
    f = SVector(
      z,z,
      g * h_upper_ll * (b_ll +   h_lower_ll) + g * h_upper_average * (b_jump +   h_lower_jump),
      z,z,
      g * h_lower_ll * (b_ll + equations.r * h_upper_ll) + g * h_lower_average * (b_jump + 
          equations.r * h_upper_jump),
      z)
  end

  return f
end

@inline function flux_nonconservative_fjordholm_etal(u_ll, u_rr,
                                                     normal_direction_ll::AbstractVector,
                                                     normal_direction_average::AbstractVector,
                                                     equations::ShallowWaterTwoLayerEquations2D)
  # Pull the necessary left and right state information
  h_upper_ll, h_v1_upper_ll, h_v2_upper_ll, h_lower_ll, h_v1_lower_ll, h_v2_lower_ll, b_ll = u_ll
  h_upper_rr, h_v1_upper_rr, h_v2_upper_rr, h_lower_rr, h_v1_lower_rr, h_v2_lower_rr, b_rr = u_rr

  # Create average and jump values
  h_upper_average = 0.5 * (h_upper_ll + h_upper_rr)
  h_lower_average = 0.5 * (h_lower_ll + h_lower_rr)
  h_upper_jump    = h_upper_rr - h_upper_ll
  h_lower_jump    = h_lower_rr - h_lower_ll
  b_jump          = b_rr  - b_ll

  # Comes in two parts:
  #   (i)  Diagonal (consistent) term from the volume flux that uses `normal_direction_average`
  #        but we use `b_ll` to avoid cross-averaging across a discontinuous bottom topography
  f2 = normal_direction_average[1] * equations.gravity*h_upper_ll*(b_ll +     h_lower_ll)
  f3 = normal_direction_average[2] * equations.gravity*h_upper_ll*(b_ll +     h_lower_ll)
  f5 = normal_direction_average[1] * equations.gravity*h_lower_ll*(b_ll + equations.r * h_upper_ll)
  f6 = normal_direction_average[2] * equations.gravity*h_lower_ll*(b_ll + equations.r * h_upper_ll)
  #   (ii) True surface part that uses `normal_direction_ll`, `h_average` and `b_jump`
  #        to handle discontinuous bathymetry
  f2 += normal_direction_ll[1] * equations.gravity*h_upper_average*(b_jump +     h_lower_jump)
  f3 += normal_direction_ll[2] * equations.gravity*h_upper_average*(b_jump +     h_lower_jump)
  f5 += normal_direction_ll[1] * equations.gravity*h_lower_average*(b_jump + 
                                                                    equations.r * h_upper_jump)
  f6 += normal_direction_ll[2] * equations.gravity*h_lower_average*(b_jump +
                                                                    equations.r * h_upper_jump)

  # Continuity equations do not have a nonconservative flux
  f1 = f4 = zero(eltype(u_ll))

return SVector(f1, f2, f3, f4, f5, f6, zero(eltype(u_ll)))
end


"""
    flux_fjordholm_etal(u_ll, u_rr, orientation,
                        equations::ShallowWaterTwoLayerEquations2D)

Total energy conservative (mathematical entropy for two-layer shallow water equations). When the 
bottom topography is nonzero this should only be used as a surface flux otherwise the scheme will 
not be well-balanced. For well-balancedness in the volume flux use [`flux_wintermeyer_etal`](@ref).

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
                                     equations::ShallowWaterTwoLayerEquations2D)
  # Unpack left and right state
  h_upper_ll, h_lower_ll = waterheight(u_ll, equations)
  v1_upper_ll, v2_upper_ll, v1_lower_ll, v2_lower_ll = velocity(u_ll, equations)
  h_upper_rr, h_lower_rr = waterheight(u_rr, equations)
  v1_upper_rr, v2_upper_rr, v1_lower_rr, v2_lower_rr = velocity(u_rr, equations)

  # Average each factor of products in flux
  h_upper_avg  = 0.5 * (h_upper_ll  + h_upper_rr )
  h_lower_avg  = 0.5 * (h_lower_ll  + h_lower_rr )
  v1_upper_avg = 0.5 * (v1_upper_ll + v1_upper_rr )
  v1_lower_avg = 0.5 * (v1_lower_ll + v1_lower_rr )
  v2_upper_avg = 0.5 * (v2_upper_ll + v2_upper_rr )
  v2_lower_avg = 0.5 * (v2_lower_ll + v2_lower_rr )
  p1_avg = 0.25 * equations.gravity * (h_upper_ll^2 + h_upper_rr^2)
  p2_avg = 0.25 * equations.gravity * (h_lower_ll^2 + h_lower_rr^2)

  # Calculate fluxes depending on orientation
  if orientation == 1
    f1 = h_upper_avg * v1_upper_avg
    f2 = f1 * v1_upper_avg + p1_avg
    f3 = f1 * v2_upper_avg
    f4 = h_lower_avg * v1_lower_avg
    f5 = f4 * v1_lower_avg + p2_avg
    f6 = f4 * v2_lower_avg
  else
    f1 = h_upper_avg * v2_upper_avg
    f2 = f1 * v1_upper_avg
    f3 = f1 * v2_upper_avg + p1_avg
    f4 = h_lower_avg * v2_lower_avg
    f5 = f4 * v1_lower_avg
    f6 = f4 * v2_lower_avg + p2_avg
  end

  return SVector(f1, f2, f3, f4, f5, f6, zero(eltype(u_ll)))
end

@inline function flux_fjordholm_etal(u_ll, u_rr,
                                     normal_direction::AbstractVector,
                                     equations::ShallowWaterTwoLayerEquations2D)
  # Unpack left and right state
  h_upper_ll, h_lower_ll = waterheight(u_ll, equations)
  v1_upper_ll, v2_upper_ll, v1_lower_ll, v2_lower_ll = velocity(u_ll, equations)
  h_upper_rr, h_lower_rr = waterheight(u_rr, equations)
  v1_upper_rr, v2_upper_rr, v1_lower_rr, v2_lower_rr = velocity(u_rr, equations)

  # Compute velocity in normal direction
  v_upper_dot_n_ll = v1_upper_ll * normal_direction[1] + v2_upper_ll * normal_direction[2]
  v_upper_dot_n_rr = v1_upper_rr * normal_direction[1] + v2_upper_rr * normal_direction[2]
  v_lower_dot_n_ll = v1_lower_ll * normal_direction[1] + v2_lower_ll * normal_direction[2]
  v_lower_dot_n_rr = v1_lower_rr * normal_direction[1] + v2_lower_rr * normal_direction[2]

  # Average each factor of products in flux
  h_upper_avg  = 0.5 * (h_upper_ll   + h_upper_rr )
  h_lower_avg  = 0.5 * (h_lower_ll   + h_lower_rr )
  v1_upper_avg = 0.5 * (v1_upper_ll   + v1_upper_rr )
  v1_lower_avg = 0.5 * (v1_lower_ll   + v1_lower_rr )
  v2_upper_avg = 0.5 * (v2_upper_ll   + v2_upper_rr )
  v2_lower_avg = 0.5 * (v2_lower_ll   + v2_lower_rr )
  p1_avg = 0.25* equations.gravity * (h_upper_ll^2 + h_upper_rr^2)
  p2_avg = 0.25* equations.gravity * (h_lower_ll^2 + h_lower_rr^2)
  v_upper_dot_n_avg = 0.5 * (v_upper_dot_n_ll + v_upper_dot_n_rr)
  v_lower_dot_n_avg = 0.5 * (v_lower_dot_n_ll + v_lower_dot_n_rr)

  # Calculate fluxes depending on normal_direction
  f1 = h_upper_avg * v_upper_dot_n_avg
  f2 = f1 * v1_upper_avg + p1_avg * normal_direction[1]
  f3 = f1 * v2_upper_avg + p1_avg * normal_direction[2]
  f4 = h_lower_avg * v_lower_dot_n_avg
  f5 = f4 * v1_lower_avg + p2_avg * normal_direction[1]
  f6 = f4 * v2_lower_avg + p2_avg * normal_direction[2]

  return SVector(f1, f2, f3, f4, f5, f6, zero(eltype(u_ll)))
end


"""
    flux_wintermeyer_etal(u_ll, u_rr, orientation,
                          equations::ShallowWaterTwoLayerEquations2D)
                          
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
                                       equations::ShallowWaterTwoLayerEquations2D)
  # Unpack left and right state
  h_upper_ll, h_v1_upper_ll, h_v2_upper_ll, h_lower_ll, h_v1_lower_ll, h_v2_lower_ll, _ = u_ll
  h_upper_rr, h_v1_upper_rr, h_v2_upper_rr, h_lower_rr, h_v1_lower_rr, h_v2_lower_rr, _ = u_rr

  # Get the velocities on either side
  v1_upper_ll, v2_upper_ll, v1_lower_ll, v2_lower_ll = velocity(u_ll, equations)
  v1_upper_rr, v2_upper_rr, v1_lower_rr, v2_lower_rr = velocity(u_rr, equations)

  # Average each factor of products in flux
  v1_upper_avg = 0.5 * (v1_upper_ll + v1_upper_rr )
  v1_lower_avg = 0.5 * (v1_lower_ll + v1_lower_rr )
  v2_upper_avg = 0.5 * (v2_upper_ll + v2_upper_rr )
  v2_lower_avg = 0.5 * (v2_lower_ll + v2_lower_rr )
  p1_avg = 0.5 * equations.gravity * h_upper_ll * h_upper_rr
  p2_avg = 0.5 * equations.gravity * h_lower_ll * h_lower_rr

  # Calculate fluxes depending on orientation
  if orientation == 1
    f1 = 0.5 * (h_v1_upper_ll + h_v1_upper_rr)
    f2 = f1 * v1_upper_avg + p1_avg
    f3 = f1 * v2_upper_avg
    f4 = 0.5 * (h_v1_lower_ll + h_v1_lower_rr)
    f5 = f4 * v1_lower_avg + p2_avg
    f6 = f4 * v2_lower_avg
  else
    f1 = 0.5 * (h_v2_upper_ll + h_v2_upper_rr)
    f2 = f1 * v1_upper_avg
    f3 = f1 * v2_upper_avg + p1_avg
    f4 = 0.5 * (h_v2_lower_ll + h_v2_lower_rr)
    f5 = f4 * v1_lower_avg
    f6 = f4 * v2_lower_avg + p2_avg
  end

  return SVector(f1, f2, f3, f4, f5, f6, zero(eltype(u_ll)))
end

@inline function flux_wintermeyer_etal(u_ll, u_rr,
                                       normal_direction::AbstractVector,
                                       equations::ShallowWaterTwoLayerEquations2D)
  # Unpack left and right state
  h_upper_ll, h_v1_upper_ll, h_v2_upper_ll, h_lower_ll, h_v1_lower_ll, h_v2_lower_ll, _ = u_ll
  h_upper_rr, h_v1_upper_rr, h_v2_upper_rr, h_lower_rr, h_v1_lower_rr, h_v2_lower_rr, _ = u_rr

  # Get the velocities on either side
  v1_upper_ll, v2_upper_ll, v1_lower_ll, v2_lower_ll = velocity(u_ll, equations)
  v1_upper_rr, v2_upper_rr, v1_lower_rr, v2_lower_rr = velocity(u_rr, equations)

  # Average each factor of products in flux
  v1_upper_avg = 0.5 * (v1_upper_ll + v1_upper_rr )
  v1_lower_avg = 0.5 * (v1_lower_ll + v1_lower_rr )
  v2_upper_avg = 0.5 * (v2_upper_ll + v2_upper_rr )
  v2_lower_avg = 0.5 * (v2_lower_ll + v2_lower_rr )
  p1_avg = 0.5 * equations.gravity * h_upper_ll * h_upper_rr
  p2_avg = 0.5 * equations.gravity * h_lower_ll * h_lower_rr
  h_v1_upper_avg = 0.5 * (h_v1_upper_ll + h_v1_upper_rr )
  h_v2_upper_avg = 0.5 * (h_v2_upper_ll + h_v2_upper_rr )
  h_v1_lower_avg = 0.5 * (h_v1_lower_ll + h_v1_lower_rr )
  h_v2_lower_avg = 0.5 * (h_v2_lower_ll + h_v2_lower_rr )

  # Calculate fluxes depending on normal_direction
  f1 = h_v1_upper_avg * normal_direction[1] + h_v2_upper_avg * normal_direction[2]
  f2 = f1 * v1_upper_avg + p1_avg * normal_direction[1]
  f3 = f1 * v2_upper_avg + p1_avg * normal_direction[2]
  f4 = h_v1_lower_avg * normal_direction[1] + h_v2_lower_avg * normal_direction[2]
  f5 = f4 * v1_lower_avg + p2_avg * normal_direction[1]
  f6 = f4 * v2_lower_avg + p2_avg * normal_direction[2]

  return SVector(f1, f2, f3, f4, f5, f6, zero(eltype(u_ll)))
end


"""
    flux_es_fjordholm_etal(u_ll, u_rr, orientation_or_normal_direction,
                           equations::ShallowWaterTwoLayerEquations1D)
Entropy stable surface flux for the two-layer shallow water equations. Uses the entropy conservative 
flux_fjordholm_etal and adds a Lax-Friedrichs type dissipation dependent on the jump of entropy
variables. 

Further details are available in the paper:
- Ulrik Skre Fjordholm (2012)
Energy conservative and stable schemes for the two-layer shallow water equations.
[DOI: 10.1142/9789814417099_0039](https://doi.org/10.1142/9789814417099_0039)
It should be noted that the equations are ordered differently and the
designation of the upper and lower layer has been changed which leads to a slightly different
formulation.
"""
@inline function flux_es_fjordholm_etal(u_ll, u_rr,
                                        orientation_or_normal_direction, 
                                        equations::ShallowWaterTwoLayerEquations2D)   
  # Compute entropy conservative flux but without the bottom topography
  f_ec = flux_fjordholm_etal(u_ll, u_rr,
                            orientation_or_normal_direction,
                            equations)

  # Get maximum signal velocity
  λ = max_abs_speed_naive(u_ll, u_rr, orientation_or_normal_direction, equations)

  # Get entropy variables but without the bottom topography
  q_rr = cons2entropy(u_rr,equations)
  q_ll = cons2entropy(u_ll,equations)

  # Average values from left and right 
  u_avg = (u_ll + u_rr)/2

  # Introduce variables for better readability
  rho_upper = equations.rho_upper
  rho_lower = equations.rho_lower
  g    = equations.gravity
  drho = rho_upper - rho_lower

  # Entropy Jacobian matrix
  H = @SMatrix [
    [-rho_lower/(g*rho_upper*drho);;
     -rho_lower*u_avg[2]/(g*rho_upper*u_avg[1]*drho);;
     -rho_lower*u_avg[3]/(g*rho_upper*u_avg[1]*drho);;
     1.0/(g*drho);;
     u_avg[5]/(g*u_avg[4]*drho);;
     u_avg[6]/(g*u_avg[4]*drho);;
     0];
    [-rho_lower*u_avg[2]/(g*rho_upper*u_avg[1]*drho);;
     (g*rho_upper*u_avg[1]^3 - g*rho_lower*u_avg[1]^3 +
         -rho_lower*u_avg[2]^2)/(g*rho_upper*u_avg[1]^2*drho);;
     -rho_lower*u_avg[2]*u_avg[3]/(g*rho_upper*u_avg[1]^2*drho);;
     u_avg[2]/(g*u_avg[1]*drho);;
     u_avg[2]*u_avg[5]/(g*u_avg[1]*u_avg[4]*drho);;
     u_avg[2]*u_avg[6]/(g*u_avg[1]*u_avg[4]*drho);;
     0];
    [-rho_lower*u_avg[3]/(g*rho_upper*u_avg[1]*drho);;
     -rho_lower*u_avg[2]*u_avg[3]/(g*rho_upper*u_avg[1]^2*drho);;
     (g*rho_upper*u_avg[1]^3 - g*rho_lower*u_avg[1]^3 +
         -rho_lower*u_avg[3]^2)/(g*rho_upper*u_avg[1]^2*drho);;
     u_avg[3]/(g*u_avg[1]*drho);;
     u_avg[3]*u_avg[5]/(g*u_avg[1]*u_avg[4]*drho);;
     u_avg[3]*u_avg[6]/(g*u_avg[1]*u_avg[4]*drho);;
     0];
    [1.0/(g*drho);;
     u_avg[2]/(g*u_avg[1]*drho);;
     u_avg[3]/(g*u_avg[1]*drho);;
     -1.0/(g*drho);;
     -u_avg[5]/(g*u_avg[4]*drho);;
     -u_avg[6]/(g*u_avg[4]*drho);;
     0];
    [u_avg[5]/(g*u_avg[4]*drho);;
     u_avg[2]*u_avg[5]/(g*u_avg[1]*u_avg[4]*drho);;
     u_avg[3]*u_avg[5]/(g*u_avg[1]*u_avg[4]*drho);;
     -u_avg[5]/(g*u_avg[4]*drho);;
     (g*rho_upper*u_avg[4]^3 - g*rho_lower*u_avg[4]^3 +
         -rho_lower*u_avg[5]^2)/(g*rho_lower*u_avg[4]^2*drho);;
     -u_avg[5]*u_avg[6]/(g*u_avg[4]^2*drho);;
     0];
    [u_avg[6]/(g*u_avg[4]*drho);;
     u_avg[2]*u_avg[6]/(g*u_avg[1]*u_avg[4]*drho);;
     u_avg[3]*u_avg[6]/(g*u_avg[1]*u_avg[4]*drho);;
     -u_avg[6]/(g*u_avg[4]*drho);;
     -u_avg[5]*u_avg[6]/(g*u_avg[4]^2*drho);;
     (g*rho_upper*u_avg[4]^3 - g*rho_lower*u_avg[4]^3 +
     -rho_lower*u_avg[6]^2)/(g*rho_lower*u_avg[4]^2*drho);;0];
    [0;;0;;0;;0;;0;;0;;0]]

  # Add dissipation to entropy conservative flux to obtain entropy stable flux
  f_es = f_ec - 0.5 * λ * H * (q_rr - q_ll)
  
  return SVector(f_es[1], f_es[2], f_es[3], f_es[4], f_es[5], f_es[6], zero(eltype(u_ll)))
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
                                     equations::ShallowWaterTwoLayerEquations2D)
  # Unpack left and right state
  h_upper_ll, h_v1_upper_ll, h_v2_upper_ll, h_lower_ll, h_v1_lower_ll, h_v2_lower_ll, _ = u_ll
  h_upper_rr, h_v1_upper_rr, h_v2_upper_rr, h_lower_rr, h_v1_lower_rr, h_v2_lower_rr, _ = u_rr

  # Calculate averaged velocity of both layers
  if orientation == 1
    v_m_ll = (h_v1_upper_ll + h_v1_lower_ll) / (h_upper_ll + h_lower_ll)
    v_m_rr = (h_v1_upper_rr + h_v1_lower_rr) / (h_upper_rr + h_lower_rr)
  else
    v_m_ll = (h_v2_upper_ll + h_v2_lower_ll) / (h_upper_ll + h_lower_ll)
    v_m_rr = (h_v2_upper_rr + h_v2_lower_rr) / (h_upper_rr + h_lower_rr)
  end

  # Calculate the wave celerity on the left and right
  h_upper_ll, h_lower_ll = waterheight(u_ll, equations)
  h_upper_rr, h_lower_rr = waterheight(u_rr, equations)

  c_ll = sqrt(equations.gravity * (h_upper_ll + h_lower_ll) )
  c_rr = sqrt(equations.gravity * (h_upper_rr + h_lower_rr))

  return (max(abs(v_m_ll),abs(v_m_rr)) + max(c_ll,c_rr))
end


@inline function max_abs_speed_naive(u_ll, u_rr, 
                                     normal_direction::AbstractVector,
                                     equations::ShallowWaterTwoLayerEquations2D)
  # Unpack left and right state
  h_upper_ll, _, _, h_lower_ll, _, _, _ = u_ll
  h_upper_rr, _, _, h_lower_rr, _, _, _ = u_rr

  # Extract and compute the velocities in the normal direction
  v1_upper_ll, v2_upper_ll, v1_lower_ll, v2_lower_ll = velocity(u_ll, equations)
  v1_upper_rr, v2_upper_rr, v1_lower_rr, v2_lower_rr = velocity(u_rr, equations)

  v_upper_dot_n_ll = v1_upper_ll * normal_direction[1] + v2_upper_ll * normal_direction[2]
  v_upper_dot_n_rr = v1_upper_rr * normal_direction[1] + v2_upper_rr * normal_direction[2]
  v_lower_dot_n_ll = v1_lower_ll * normal_direction[1] + v2_lower_ll * normal_direction[2]
  v_lower_dot_n_rr = v1_lower_rr * normal_direction[1] + v2_lower_rr * normal_direction[2]
  
  # Calculate averaged velocity of both layers
  v_m_ll = (v_upper_dot_n_ll * h_upper_ll + v_lower_dot_n_ll * h_lower_ll) / (h_upper_ll + h_lower_ll)
  v_m_rr = (v_upper_dot_n_rr * h_upper_rr + v_lower_dot_n_rr * h_lower_rr) / (h_upper_rr + h_lower_rr)

  # Compute the wave celerity on the left and right
  h_upper_ll, h_lower_ll = waterheight(u_ll, equations)
  h_upper_rr, h_lower_rr = waterheight(u_rr, equations)

  c_ll = sqrt(equations.gravity * (h_upper_ll + h_lower_ll))
  c_rr = sqrt(equations.gravity * (h_upper_rr + h_lower_rr))

  # The normal velocities are already scaled by the norm
  return max(abs(v_m_ll), abs(v_m_rr)) + max(c_ll, c_rr) * norm(normal_direction)
end


# Specialized `DissipationLocalLaxFriedrichs` to avoid spurious dissipation in the bottom topography
@inline function (dissipation::DissipationLocalLaxFriedrichs)(u_ll, u_rr, 
    orientation_or_normal_direction, equations::ShallowWaterTwoLayerEquations2D)
  λ = dissipation.max_abs_speed(u_ll, u_rr, orientation_or_normal_direction, equations)
  diss = -0.5 * λ * (u_rr - u_ll)
  return SVector(diss[1], diss[2], diss[3], diss[4], diss[5], diss[6], zero(eltype(u_ll)))
end


# Absolute speed of the barotropic mode
@inline function max_abs_speeds(u, equations::ShallowWaterTwoLayerEquations2D)
  h_upper, h_v1_upper, h_v2_upper, h_lower, h_v1_lower, h_v2_lower, _ = u

  # Calculate averaged velocity of both layers
  v1_m = (h_v1_upper + h_v1_lower) / (h_upper + h_lower)
  v2_m = (h_v2_upper + h_v2_lower) / (h_upper + h_lower)

  h_upper, h_lower = waterheight(u, equations)
  v1_upper, v2_upper, v1_lower, v2_lower = velocity(u, equations)

  c = sqrt(equations.gravity * (h_upper + h_lower)) 
  return (max(abs(v1_m) + c, abs(v1_upper), abs(v1_lower)), 
          max(abs(v2_m) + c, abs(v2_upper), abs(v2_lower)))
end


# Helper function to extract the velocity vector from the conservative variables
@inline function velocity(u, equations::ShallowWaterTwoLayerEquations2D)
  h_upper, h_v1_upper, h_v2_upper, h_lower, h_v1_lower, h_v2_lower, _ = u

  v1_upper = h_v1_upper / h_upper
  v2_upper = h_v2_upper / h_upper
  v1_lower = h_v1_lower / h_lower
  v2_lower = h_v2_lower / h_lower

  return SVector(v1_upper, v2_upper, v1_lower, v2_lower)
end


# Convert conservative variables to primitive
@inline function cons2prim(u, equations::ShallowWaterTwoLayerEquations2D)
  h_upper, _, _, h_lower, _, _, b = u

  H_lower = h_lower + b
  H_upper = h_lower + h_upper + b
  v1_upper, v2_upper, v1_lower, v2_lower = velocity(u, equations)

  return SVector(H_upper, v1_upper, v2_upper , H_lower, v1_lower, v2_lower, b)
end


# Convert conservative variables to entropy variables
# Note, only the first four are the entropy variables, the fifth entry still just carries the bottom
# topography values for convenience. 
# In contrast to general usage the entropy variables are denoted with q instead of w, because w is
# already used for velocity in y-Direction
@inline function cons2entropy(u, equations::ShallowWaterTwoLayerEquations2D)
  h_upper, _, _, h_lower, _, _, b = u
  # Assign new variables for better readability
  rho_upper = equations.rho_upper
  rho_lower = equations.rho_lower
  v1_upper, v2_upper, v1_lower, v2_lower = velocity(u, equations)

  w1 = rho_upper * (equations.gravity * (              h_upper + h_lower + b) +
               - 0.5 * (v1_upper^2 + v2_upper^2))
  w2 = rho_upper * v1_upper
  w3 = rho_upper * v2_upper
  w4 = rho_lower * (equations.gravity * (equations.r * h_upper + h_lower + b) +
               - 0.5 * (v1_lower^2 + v2_lower^2))
  w5 = rho_lower * v1_lower
  w6 = rho_lower * v2_lower
  return SVector(w1, w2, w3, w4, w5, w6, b)
end


# Convert primitive to conservative variables
@inline function prim2cons(prim, equations::ShallowWaterTwoLayerEquations2D)
  H_upper, v1_upper, v2_upper, H_lower, v1_lower, v2_lower, b = prim

  h_lower = H_lower - b
  h_upper = H_upper - h_lower - b
  h_v1_upper = h_upper * v1_upper
  h_v2_upper = h_upper * v2_upper
  h_v1_lower = h_lower * v1_lower
  h_v2_lower = h_lower * v2_lower
  return SVector(h_upper, h_v1_upper, h_v2_upper, h_lower, h_v1_lower, h_v2_lower, b)
end


@inline function waterheight(u, equations::ShallowWaterTwoLayerEquations2D)
  return SVector(u[1], u[4])
end


# Entropy function for the shallow water equations is the total energy
@inline entropy(cons, equations::ShallowWaterTwoLayerEquations2D) = energy_total(cons, equations)


# Calculate total energy for a conservative state `cons`
@inline function energy_total(cons, equations::ShallowWaterTwoLayerEquations2D)
  h_upper, h_v1_upper, h_v2_upper, h_lower, h_v2_lower, h_v2_lower, b = cons
  g = equations.gravity
  rho_upper= equations.rho_upper
  rho_lower= equations.rho_lower

  e = (0.5 * rho_upper * (h_v1_upper^2 / h_upper + h_v2_upper^2 / h_upper + g * h_upper^2) +
       0.5 * rho_lower * (h_v2_lower^2 / h_lower + h_v2_lower^2 / h_lower + g * h_lower^2) + 
       g*rho_lower*h_lower*b + g*rho_upper*h_upper*(h_lower + b))
  return e
end


# Calculate kinetic energy for a conservative state `cons`
@inline function energy_kinetic(u, equations::ShallowWaterTwoLayerEquations2D)
  h_upper, h_v1_upper, h_v2_upper, h_lower, h_v2_lower, h_v2_lower, _ = u

  return (0.5 * equations.rho_upper * h_v1_upper^2 / h_upper +
          0.5 * equations.rho_upper * h_v2_upper^2 / h_upper +
          0.5 * equations.rho_lower * h_v2_lower^2 / h_lower +
          0.5 * equations.rho_lower * h_v2_lower^2 / h_lower)
end


# Calculate potential energy for a conservative state `cons`
@inline function energy_internal(cons, equations::ShallowWaterTwoLayerEquations2D)
  return energy_total(cons, equations) - energy_kinetic(cons, equations)
end


# Calculate the error for the "lake-at-rest" test case where H = h_upper+h_lower+b should
# be a constant value over time
@inline function lake_at_rest_error(u, equations::ShallowWaterTwoLayerEquations2D)
  h_upper, _, _, h_lower, _, _, b = u
  return abs(equations.H0 - (h_upper + h_lower + b))
end

end # @muladd
