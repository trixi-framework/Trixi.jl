# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin


@doc raw"""
    TwoLayerShallowWaterEquations2D(gravity, H0, rho1, rho2)

Two-Layer Shallow water equations (2LSWE) in two space dimension. The equations are given by
```math
\begin{alignat}{7}
& (h_1)_t        &+& (h_1v_1)_x &+& (h_1w_1)_y &=& 0 \\
& (h_1v_1)_t  &+& (h_1v_1^2 + \frac{gh_1^2}{2})_x &+& (h_1v_1w_1)_y &=& -gh_1(b+h_2)_x \\
& (h_1w_1)_t &+& (h_1v_1w_1)_x &+& (h_1w_1^2 + \tfrac{gh_1^2}{2})_y &=& -gh_1(b+h_2)_y\\
& (h_2)_t       &+& (h_2v_2)_x &+& (h_2w_2)_y &=& 0 \\
& (h_2v_2)_t  &+& (h_2v_2^2 + \dfrac{gh_2^2}{2})_x &+& (h_2v_2w_2)_y &=
& -gh_2(b+\dfrac{\rho_1}{\rho_2} h_1)_x\\
& (h_2w_2)_t  &+& (h_2v_2w_2)_x &+& (h_2w_2^2 + \dfrac{gh_2^2}{2})_y &=
& -gh_2(b+\dfrac{\rho_1}{\rho_2} h_1)_y
\end{alignat}
```
The unknown quantities of the SWE are the water heights of the lower layer ``h_2`` and the upper 
layer ``h_1`` and the respecitve velocities in x-Direction ``v_1`` and ``v_2`` and in y-Direction
``w_1`` and ``w_2``. The gravitational constant is denoted by `g`, the layer densitites by 
``\rho_1``and ``\rho_2`` and the (possibly) variable bottom topography function by ``b(x)``. 
Conservative variable water height ``h_2`` is measured from the bottom topography ``b`` and ``h_1`` 
relative to ``h_2``, therefore one also defines the total water heights as ``H2 = h2 + b`` and 
``H1 = h1 + h2 + b``.

The additional quantity ``H_0`` is also available to store a reference value for the total water
height that is useful to set initial conditions or test the "lake-at-rest" well-balancedness.

The bottom topography function ``b(x)`` is set inside the initial condition routine
for a particular problem setup. To test the conservative form of the 2LSWE one can set the bottom 
topography variable `b` to zero.

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

A good introduction for the two-layer SWE is available in Chapter 12 of the book:
  - Benoit Cushman-Roisin (2011)
    Introduction to geophyiscal fluid dynamics: physical and numerical aspects
    @link https://www.sciencedirect.com/bookseries/international-geophysics/vol/101/suppl/C
    ISBN: 978-0-12-088759-0
"""
struct TwoLayerShallowWaterEquations2D{RealT<:Real} <: AbstractShallowWaterEquations{2, 7}
  gravity::RealT # gravitational constant
  H0::RealT      # constant "lake-at-rest" total water height
  rho1::RealT    # lower layer density
  rho2::RealT    # upper layer density
  r::RealT       # # ratio of rho1 / rho2
end

# Allow for flexibility to set the gravitational constant within an elixir depending on the
# application where `gravity_constant=1.0` or `gravity_constant=9.81` are common values.
# The reference total water height H0 defaults to 0.0 but is used for the "lake-at-rest"
# well-balancedness test cases. Densities must be specificed such that rho_1 < rho_2.
function TwoLayerShallowWaterEquations2D(; gravity_constant, H0=0.0, rho1, rho2)
  r = rho1 / rho2
  TwoLayerShallowWaterEquations2D(gravity_constant, H0, rho1, rho2, r)
end


have_nonconservative_terms(::TwoLayerShallowWaterEquations2D) = True()
varnames(::typeof(cons2cons), ::TwoLayerShallowWaterEquations2D) = ("h1", "h1_v1", "h1_w1", "h2",
                                                                    "h2_v2", "h2_w2", "b")                                                             
# Note, we use the total water height, H1 = h1 + h2 + b, and first layer total heigth H2 = h2 + b as
# the first primitive variable for easier visualization and setting initial conditions
varnames(::typeof(cons2prim), ::TwoLayerShallowWaterEquations2D) = ("H1", "v1", "w1", "H2", "v2",
                                                                    "w2", "b")


# Set initial conditions at physical location `x` for time `t`
"""
    initial_condition_convergence_test(x, t, equations::ShallowWaterEquations2D)

A smooth initial condition used for convergence tests in combination with
[`source_terms_convergence_test`](@ref). Constants must be set to ρ_1 = 0.9, ρ_2 = 1.0, g = 10.0.
"""
function initial_condition_convergence_test(x, t, equations::TwoLayerShallowWaterEquations2D)
  # some constants are chosen such that the function is periodic on the domain [0,sqrt(2)]^2]
  ω = 2.0 * pi * sqrt(2.0)

  H2 = 2.0 + 0.1 * sin(ω * x[1] + t) * cos(ω * x[2] + t)
  H1 = 4.0 + 0.1 * cos(ω * x[1] + t) * sin(ω * x[2] + t)
  v2 = 1.0
  v1 = 0.9
  w2 = 0.9
  w1 = 1.0
  b  = 1.0 + 0.1 * cos(0.5 * ω * x[1]) * sin(0.5 * ω * x[2])

  return prim2cons(SVector(H1, v1, w1, H2, v2, w2, b), equations)
end


"""
    source_terms_convergence_test(u, x, t, equations::ShallowWaterEquations2D)

Source terms used for convergence tests in combination with
[`initial_condition_convergence_test`](@ref).
"""
@inline function source_terms_convergence_test(u, x, t, equations::TwoLayerShallowWaterEquations2D)
  # Same settings as in `initial_condition_convergence_test`. 
  # some constants are chosen such that the function is periodic on the domain [0,sqrt(2)]^2]
  ω = 2.0 * pi * sqrt(2.0)


  # Source terms obtained with SymPy
  du1 = 0.01ω*cos(t + ω*x[1])*cos(t + ω*x[2]) + 0.01ω*sin(t + ω*x[1])*sin(t + ω*x[2])
  du2 = (5.0(-0.1ω*cos(t + ω*x[1])*cos(t + ω*x[2]) - 0.1ω*sin(t + ω*x[1])*sin(t + ω*x[2]))*(4.0 +
         0.2cos(t + ω*x[1])*sin(t + ω*x[2]) - 0.2sin(t + ω*x[1])*cos(t + ω*x[2])) + 0.009ω*cos(t +
         ω*x[1])*cos(t + ω*x[2]) + 0.009ω*sin(t + ω*x[1])*sin(t + ω*x[2]) + 0.1ω*(20.0 + cos(t +
         ω*x[1])*sin(t + ω*x[2]) - sin(t + ω*x[1])*cos(t + ω*x[2]))*cos(t + ω*x[1])*cos(t + ω*x[2]))
  du3 = (5.0(0.1ω*cos(t + ω*x[1])*cos(t + ω*x[2]) + 0.1ω*sin(t + ω*x[1])*sin(t + ω*x[2]))*(4.0 +
         0.2cos(t + ω*x[1])*sin(t + ω*x[2]) - 0.2sin(t + ω*x[1])*cos(t + ω*x[2])) + 0.01ω*cos(t +
         ω*x[1])*cos(t + ω*x[2]) + 0.01ω*sin(t + ω*x[1])*sin(t + ω*x[2]) - 0.1ω*(20.0 + cos(t +
         ω*x[1])*sin(t + ω*x[2]) - sin(t + ω*x[1])*cos(t + ω*x[2]))*sin(t + ω*x[1])*sin(t + ω*x[2]))
  du4 = (0.1cos(t + ω*x[1])*cos(t + ω*x[2]) + 0.1ω*cos(t + ω*x[1])*cos(t + ω*x[2]) +
        0.05ω*sin(0.5ω*x[1])*sin(0.5ω*x[2]) - 0.1sin(t + ω*x[1])*sin(t + ω*x[2]) +
        -0.045ω*cos(0.5ω*x[1])*cos(0.5ω*x[2]) - 0.09ω*sin(t + ω*x[1])*sin(t + ω*x[2]))
  du5 = ((10.0 + sin(t + ω*x[1])*cos(t + ω*x[2]) - cos(0.5ω*x[1])*sin(0.5ω*x[2]))*(-0.09ω*cos(t +
         ω*x[1])*cos(t + ω*x[2]) - 0.09ω*sin(t + ω*x[1])*sin(t + ω*x[2]) +
         -0.05ω*sin(0.5ω*x[1])*sin(0.5ω*x[2])) + 5.0(0.1ω*cos(t + ω*x[1])*cos(t + ω*x[2]) +
         0.05ω*sin(0.5ω*x[1])*sin(0.5ω*x[2]))*(2.0 + 0.2sin(t + ω*x[1])*cos(t + ω*x[2]) +
         -0.2cos(0.5ω*x[1])*sin(0.5ω*x[2])) + 0.1cos(t + ω*x[1])*cos(t + ω*x[2]) + 0.1ω*cos(t +
         ω*x[1])*cos(t + ω*x[2]) + 0.05ω*sin(0.5ω*x[1])*sin(0.5ω*x[2]) - 0.1sin(t + ω*x[1])*sin(t +
         ω*x[2]) - 0.045ω*cos(0.5ω*x[1])*cos(0.5ω*x[2]) - 0.09ω*sin(t + ω*x[1])*sin(t + ω*x[2]))
  du6 = ((10.0 + sin(t + ω*x[1])*cos(t + ω*x[2]) +
          -cos(0.5ω*x[1])*sin(0.5ω*x[2]))*(0.05ω*cos(0.5ω*x[1])*cos(0.5ω*x[2]) + 0.09ω*cos(t +
          ω*x[1])*cos(t + ω*x[2]) + 0.09ω*sin(t + ω*x[1])*sin(t + ω*x[2])) +
          5.0(-0.05ω*cos(0.5ω*x[1])*cos(0.5ω*x[2]) - 0.1ω*sin(t + ω*x[1])*sin(t + ω*x[2]))*(2.0 +
          0.2sin(t + ω*x[1])*cos(t + ω*x[2]) - 0.2cos(0.5ω*x[1])*sin(0.5ω*x[2])) + 0.09cos(t +
          ω*x[1])*cos(t + ω*x[2]) + 0.09ω*cos(t + ω*x[1])*cos(t + ω*x[2]) +
          0.045ω*sin(0.5ω*x[1])*sin(0.5ω*x[2]) - 0.09sin(t + ω*x[1])*sin(t + ω*x[2]) +
          -0.0405ω*cos(0.5ω*x[1])*cos(0.5ω*x[2]) - 0.081ω*sin(t + ω*x[1])*sin(t + ω*x[2]))

  return SVector(du1, du2, du3, du4, du5, du6, 0.0)
end


"""
    boundary_condition_slip_wall(u_inner, normal_direction, x, t, surface_flux_function,
                                 equations::ShallowWaterEquations2D)

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
                                              equations::TwoLayerShallowWaterEquations2D)
  # normalize the outward pointing direction
  normal = normal_direction / norm(normal_direction)

  # compute the normal velocity
  v1_normal = normal[1] * u_inner[2] + normal[2] * u_inner[3]
  v2_normal = normal[1] * u_inner[5] + normal[2] * u_inner[6]

  # create the "external" boundary solution state
  u_boundary = SVector(u_inner[1],
                       u_inner[2] - 2.0 * v1_normal * normal[1],
                       u_inner[3] - 2.0 * v1_normal * normal[2],
                       u_inner[4],
                       u_inner[5] - 2.0 * v2_normal * normal[1],
                       u_inner[6] - 2.0 * v2_normal * normal[2],
                       u_inner[7])

  # calculate the boundary flux
  flux = surface_flux_function(u_inner, u_boundary, normal_direction, equations)
  return flux
end


# Calculate 1D flux for a single point
# Note, the bottom topography has no flux
@inline function flux(u, orientation::Integer, equations::TwoLayerShallowWaterEquations2D)
  h1, h1_v1, h1_w1, h2, h2_v2, h2_w2, _ = u

  # Calculate velocities
  v1, w1, v2, w2 = velocity(u, equations)

  # Calculate pressure
  p1 = 0.5 * equations.gravity * h1^2
  p2 = 0.5 * equations.gravity * h2^2

  if orientation == 1
    f1 = h1_v1
    f2 = h1_v1 * v1 + p1
    f3 = h1_v1 * w1
    f4 = h2_v2
    f5 = h2_v2 * v2 + p2
    f6 = h2_v2 * w2
  else
    f1 = h1_w1
    f2 = h1_w1 * v1
    f3 = h1_w1 * w1 + p1
    f4 = h2_w2
    f5 = h2_w2 * v2
    f6 = h2_w2 * w2 + p2
  end
  return SVector(f1, f2, f3, f4, f5 , f6, zero(eltype(u)))
end

# Calculate 1D flux for a single point in the normal direction
# Note, this directional vector is not normalized and the bottom topography has no flux
@inline function flux(u, normal_direction::AbstractVector, 
                      equations::TwoLayerShallowWaterEquations2D)
  h1, h2 = waterheight(u, equations)
  v1, w1, v2, w2 = velocity(u, equations)

  v1_normal = v1 * normal_direction[1] + w1 * normal_direction[2]
  v2_normal = v2 * normal_direction[1] + w2 * normal_direction[2]
  h1_v1_normal = h1 * v1_normal
  h2_v2_normal = h2 * v2_normal

  p1 = 0.5 * equations.gravity * h1^2
  p2 = 0.5 * equations.gravity * h2^2

  f1 = h1_v1_normal
  f2 = h1_v1_normal * v1 + p1 * normal_direction[1]
  f3 = h1_v1_normal * w1 + p1 * normal_direction[2]
  f4 = h2_v2_normal
  f5 = h2_v2_normal * v2 + p2 * normal_direction[1]
  f6 = h2_v2_normal * w2 + p2 * normal_direction[2]

  return SVector(f1, f2, f3, f4, f5, f6, zero(eltype(u)))
end


"""
    flux_nonconservative_wintermeyer_etal(u_ll, u_rr, orientation::Integer,
                                          equations::TwoLayerShallowWaterEquations2D)

Non-symmetric two-point volume flux discretizing the nonconservative (source) term
that contains the gradient of the bottom topography [`TwoLayerShallowWaterEquations2D`](@ref). This
is a slightly modified version to account for the additional source term compared to standard SWE.

Further details are available in the paper:
- Niklas Wintermeyer, Andrew R. Winters, Gregor J. Gassner and David A. Kopriva (2017)
  An entropy stable nodal discontinuous Galerkin method for the two dimensional
  shallow water equations on unstructured curvilinear meshes with discontinuous bathymetry
  [DOI: 10.1016/j.jcp.2017.03.036](https://doi.org/10.1016/j.jcp.2017.03.036)
"""
@inline function flux_nonconservative_wintermeyer_etal(u_ll, u_rr, 
                                                       orientation::Integer,
                                                       equations::TwoLayerShallowWaterEquations2D)
  # Pull the necessary left and right state information
  h1_ll, h2_ll = waterheight(u_ll, equations)
  h1_rr, h2_rr = waterheight(u_rr, equations)
  b_rr = u_rr[7]

  z = zero(eltype(u_ll))

  # Bottom gradient nonconservative term: (0, g*h1*(b + h2)_x, g*h1*(b + h2)_y ,
  #                                        0, g*h2*(b + r*h1)_x, g*h2*(b + r*h1)_y, 0)
  if orientation == 1
    f = SVector(z,
    equations.gravity * h1_ll * (b_rr + h2_rr),
    z,z,
    equations.gravity * h2_ll * (b_rr + equations.r * h1_rr),
    z,z)
  else # orientation == 2
    f = SVector(z, z,
    equations.gravity * h1_ll * (b_rr + h2_rr),
    z,z,
    equations.gravity * h2_ll * (b_rr + equations.r * h1_rr),
    z)
  end

  return f
end

@inline function flux_nonconservative_wintermeyer_etal(u_ll, u_rr,
                                                       normal_direction_ll::AbstractVector,
                                                       normal_direction_average::AbstractVector,
                                                       equations::TwoLayerShallowWaterEquations2D)
  # Pull the necessary left and right state information
  h1_ll, h2_ll = waterheight(u_ll, equations)
  h1_rr, h2_rr = waterheight(u_rr, equations)
  b_rr = u_rr[7]

  # Note this routine only uses the `normal_direction_average` and the average of the
  # bottom topography to get a quadratic split form DG gradient on curved elements
  return SVector(zero(eltype(u_ll)),
                normal_direction_average[1] * equations.gravity * h1_ll * (b_rr +  h2_rr),
                normal_direction_average[2] * equations.gravity * h1_ll * (b_rr +  h2_rr),
                zero(eltype(u_ll)),
                normal_direction_average[1] * equations.gravity * h2_ll * (b_rr + 
                                                            equations.r * h1_rr),
                normal_direction_average[2] * equations.gravity * h2_ll * (b_rr +
                                                            equations.r * h1_rr),
                zero(eltype(u_ll)))
  end


"""
    flux_nonconservative_fjordholm_etal(u_ll, u_rr, orientation::Integer,
                                        equations::TwoLayerShallowWaterEquations2D)

Non-symmetric two-point surface flux discretizing the nonconservative (source) term that contains 
the gradients of the bottom topography and the layer heights 
[`TwoLayerShallowWaterEquations2D`](@ref).

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
                                                     equations::TwoLayerShallowWaterEquations2D)
  # Pull the necessary left and right state information
  h1_ll, h1_v1_ll, h1_w1_ll, h2_ll, h2_v2_ll, h2_w2_ll, b_ll = u_ll
  h1_rr, h1_v1_rr, h1_w1_rr, h2_rr, h2_v2_rr, h2_w2_rr, b_rr = u_rr

  # Create average and jump values
  h1_average = 0.5 * (h1_ll + h1_rr)
  h2_average = 0.5 * (h2_ll + h2_rr)
  h1_jump    = h1_rr - h1_ll
  h2_jump    = h2_rr - h2_ll
  b_jump     = b_rr  - b_ll

  # Assign variables for constants for better readability
  g = equations.gravity

  # Bottom gradient nonconservative term: (0, g*h1*(b+h2)_x  , g*h1*(b+h2)_y  , 
  #                                        0, g*h2*(b+r*h1)_x, g*h2*(b+r*h1)_x, 0)

  # Includes two parts:
  #   (i)  Diagonal (consistent) term from the volume flux that uses `b_ll` to avoid
  #        cross-averaging across a discontinuous bottom topography
  #   (ii) True surface part that uses `h_average` and `b_jump` to handle discontinuous bathymetry
  z = zero(eltype(u_ll))
  if orientation == 1 
    f = SVector(
      z,
      g * h1_ll * (b_ll +   h2_ll) + g * h1_average * (b_jump +   h2_jump),
      z,z,
      g * h2_ll * (b_ll + equations.r * h1_ll) + g * h2_average * (b_jump + equations.r * h1_jump),
      z,z)
  else # orientation == 2
    f = SVector(
      z,z,
      g * h1_ll * (b_ll +   h2_ll) + g * h1_average * (b_jump +   h2_jump),
      z,z,
      g * h2_ll * (b_ll + equations.r * h1_ll) + g * h2_average * (b_jump + equations.r * h1_jump),
      z)
  end

  return f
end

@inline function flux_nonconservative_fjordholm_etal(u_ll, u_rr,
                                                     normal_direction_ll::AbstractVector,
                                                     normal_direction_average::AbstractVector,
                                                     equations::TwoLayerShallowWaterEquations2D)
  # Pull the necessary left and right state information
  h1_ll, h1_v1_ll, h1_w1_ll, h2_ll, h2_v2_ll, h2_w2_ll, b_ll = u_ll
  h1_rr, h1_v1_rr, h1_w1_rr, h2_rr, h2_v2_rr, h2_w2_rr, b_rr = u_rr

  # Create average and jump values
  h1_average = 0.5 * (h1_ll + h1_rr)
  h2_average = 0.5 * (h2_ll + h2_rr)
  h1_jump    = h1_rr - h1_ll
  h2_jump    = h2_rr - h2_ll
  b_jump     = b_rr  - b_ll

  # Comes in two parts:
  #   (i)  Diagonal (consistent) term from the volume flux that uses `normal_direction_average`
  #        but we use `b_ll` to avoid cross-averaging across a discontinuous bottom topography
  f2 = normal_direction_average[1] * equations.gravity*h1_ll*(b_ll +     h2_ll)
  f3 = normal_direction_average[2] * equations.gravity*h1_ll*(b_ll +     h2_ll)
  f5 = normal_direction_average[1] * equations.gravity*h2_ll*(b_ll + equations.r * h1_ll)
  f6 = normal_direction_average[2] * equations.gravity*h2_ll*(b_ll + equations.r * h1_ll)
  #   (ii) True surface part that uses `normal_direction_ll`, `h_average` and `b_jump`
  #        to handle discontinuous bathymetry
  f2 += normal_direction_ll[1] * equations.gravity*h1_average*(b_jump +     h2_jump)
  f3 += normal_direction_ll[2] * equations.gravity*h1_average*(b_jump +     h2_jump)
  f5 += normal_direction_ll[1] * equations.gravity*h2_average*(b_jump + equations.r * h1_jump)
  f6 += normal_direction_ll[2] * equations.gravity*h2_average*(b_jump + equations.r * h1_jump)

  # Continuity equations do not have a nonconservative flux
  f1 = f4 = zero(eltype(u_ll))

return SVector(f1, f2, f3, f4, f5, f6, zero(eltype(u_ll)))
end


"""
    flux_fjordholm_etal(u_ll, u_rr, orientation,
                        equations::TwoLayerShallowWaterEquations2D)

Total energy conservative (mathematical entropy for two-layer shallow water equations). When the 
bottom topography is nonzero this should only be used as a surface flux otherwise the scheme will 
not be well-balanced. For well-balancedness in the volume flux use [`flux_wintermeyer_etal`](@ref).

Details are available in Eq. (4.1) in the paper:
- Ulrik S. Fjordholm, Siddhartha Mishr and Eitan Tadmor (2011)
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
                                     equations::TwoLayerShallowWaterEquations2D)
  # Unpack left and right state
  h1_ll, h2_ll = waterheight(u_ll, equations)
  v1_ll, w1_ll, v2_ll, w2_ll = velocity(u_ll, equations)
  h1_rr, h2_rr = waterheight(u_rr, equations)
  v1_rr, w1_rr, v2_rr, w2_rr = velocity(u_rr, equations)

  # Average each factor of products in flux
  h1_avg = 0.5 * (h1_ll + h1_rr )
  h2_avg = 0.5 * (h2_ll + h2_rr )
  v1_avg = 0.5 * (v1_ll + v1_rr )
  v2_avg = 0.5 * (v2_ll + v2_rr )
  w1_avg = 0.5 * (w1_ll + w1_rr )
  w2_avg = 0.5 * (w2_ll + w2_rr )
  p1_avg = 0.25* equations.gravity * (h1_ll^2 + h1_rr^2)
  p2_avg = 0.25* equations.gravity * (h2_ll^2 + h2_rr^2)

  # Calculate fluxes depending on orientation
  if orientation == 1
    f1 = h1_avg * v1_avg
    f2 = f1 * v1_avg + p1_avg
    f3 = f1 * w1_avg
    f4 = h2_avg * v2_avg
    f5 = f4 * v2_avg + p2_avg
    f6 = f4 * w2_avg
  else
    f1 = h1_avg * w1_avg
    f2 = f1 * v1_avg
    f3 = f1 * w1_avg + p1_avg
    f4 = h2_avg * w2_avg
    f5 = f4 * v2_avg
    f6 = f4 * w2_avg + p2_avg
  end

  return SVector(f1, f2, f3, f4, f5, f6, zero(eltype(u_ll)))
end

@inline function flux_fjordholm_etal(u_ll, u_rr,
                                     normal_direction::AbstractVector,
                                     equations::TwoLayerShallowWaterEquations2D)
  # Unpack left and right state
  h1_ll, h2_ll = waterheight(u_ll, equations)
  v1_ll, w1_ll, v2_ll, w2_ll = velocity(u_ll, equations)
  h1_rr, h2_rr = waterheight(u_rr, equations)
  v1_rr, w1_rr, v2_rr, w2_rr = velocity(u_rr, equations)

  # Compute velocity in normal direction
  v1_dot_n_ll = v1_ll * normal_direction[1] + w1_ll * normal_direction[2]
  v1_dot_n_rr = v1_rr * normal_direction[1] + w1_rr * normal_direction[2]
  v2_dot_n_ll = v2_ll * normal_direction[1] + w2_ll * normal_direction[2]
  v2_dot_n_rr = v2_rr * normal_direction[1] + w2_rr * normal_direction[2]

  # Average each factor of products in flux
  h1_avg = 0.5 * (h1_ll   + h1_rr )
  h2_avg = 0.5 * (h2_ll   + h2_rr )
  v1_avg = 0.5 * (v1_ll   + v1_rr )
  v2_avg = 0.5 * (v2_ll   + v2_rr )
  w1_avg = 0.5 * (w1_ll   + w1_rr )
  w2_avg = 0.5 * (w2_ll   + w2_rr )
  p1_avg = 0.25* equations.gravity * (h1_ll^2 + h1_rr^2)
  p2_avg = 0.25* equations.gravity * (h2_ll^2 + h2_rr^2)
  v1_dot_n_avg = 0.5 * (v1_dot_n_ll + v1_dot_n_rr)
  v2_dot_n_avg = 0.5 * (v2_dot_n_ll + v2_dot_n_rr)

  # Calculate fluxes depending on normal_direction
  f1 = h1_avg * v1_dot_n_avg
  f2 = f1 * v1_avg + p1_avg * normal_direction[1]
  f3 = f1 * w1_avg + p1_avg * normal_direction[2]
  f4 = h2_avg * v2_dot_n_avg
  f5 = f4 * v2_avg + p2_avg * normal_direction[1]
  f6 = f4 * w2_avg + p2_avg * normal_direction[2]

  return SVector(f1, f2, f3, f4, f5, f6, zero(eltype(u_ll)))
end


"""
    flux_wintermeyer_etal(u_ll, u_rr, orientation,
                          equations::TwoLayerShallowWaterEquations2D)

Total energy conservative (mathematical entropy for two-layer shallow water equations) split form.
When the bottom topography is nonzero this scheme will be well-balanced when used as a `volume_flux`.
The `surface_flux` should still use, e.g., [`flux_fjordholm_etal`](@ref).

Further details are available in Theorem 1 of the paper:
- Niklas Wintermeyer, Andrew R. Winters, Gregor J. Gassner and David A. Kopriva (2017)
  An entropy stable nodal discontinuous Galerkin method for the two dimensional
  shallow water equations on unstructured curvilinear meshes with discontinuous bathymetry
  [DOI: 10.1016/j.jcp.2017.03.036](https://doi.org/10.1016/j.jcp.2017.03.036)
"""
@inline function flux_wintermeyer_etal(u_ll, u_rr, orientation::Integer, 
                                                         equations::TwoLayerShallowWaterEquations2D)
  # Unpack left and right state
  h1_ll, h1_v1_ll, h1_w1_ll, h2_ll, h2_v2_ll, h2_w2_ll, _ = u_ll
  h1_rr, h1_v1_rr, h1_w1_rr, h2_rr, h2_v2_rr, h2_w2_rr, _ = u_rr

  # Get the velocities on either side
  v1_ll, w1_ll, v2_ll, w2_ll = velocity(u_ll, equations)
  v1_rr, w1_rr, v2_rr, w2_rr = velocity(u_rr, equations)

  # Average each factor of products in flux
  v1_avg = 0.5 * (v1_ll + v1_rr )
  v2_avg = 0.5 * (v2_ll + v2_rr )
  w1_avg = 0.5 * (w1_ll + w1_rr )
  w2_avg = 0.5 * (w2_ll + w2_rr )
  p1_avg = 0.5 * equations.gravity * h1_ll * h1_rr
  p2_avg = 0.5 * equations.gravity * h2_ll * h2_rr

  # Calculate fluxes depending on orientation
  if orientation == 1
    f1 = 0.5 * (h1_v1_ll + h1_v1_rr)
    f2 = f1 * v1_avg + p1_avg
    f3 = f1 * w1_avg
    f4 = 0.5 * (h2_v2_ll + h2_v2_rr)
    f5 = f4 * v2_avg + p2_avg
    f6 = f4 * w2_avg
  else
    f1 = 0.5 * (h1_w1_ll + h1_w1_rr)
    f2 = f1 * v1_avg
    f3 = f1 * w1_avg + p1_avg
    f4 = 0.5 * (h2_w2_ll + h2_w2_rr)
    f5 = f4 * v2_avg
    f6 = f4 * w2_avg + p2_avg
  end

  return SVector(f1, f2, f3, f4, f5, f6, zero(eltype(u_ll)))
end

@inline function flux_wintermeyer_etal(u_ll, u_rr,
                                       normal_direction::AbstractVector,
                                       equations::TwoLayerShallowWaterEquations2D)
  # Unpack left and right state
  h1_ll, h1_v1_ll, h1_w1_ll, h2_ll, h2_v2_ll, h2_w2_ll, _ = u_ll
  h1_rr, h1_v1_rr, h1_w1_rr, h2_rr, h2_v2_rr, h2_w2_rr, _ = u_rr

  # Get the velocities on either side
  v1_ll, w1_ll, v2_ll, w2_ll = velocity(u_ll, equations)
  v1_rr, w1_rr, v2_rr, w2_rr = velocity(u_rr, equations)

  # Average each factor of products in flux
  v1_avg = 0.5 * (v1_ll + v1_rr )
  v2_avg = 0.5 * (v2_ll + v2_rr )
  w1_avg = 0.5 * (w1_ll + w1_rr )
  w2_avg = 0.5 * (w2_ll + w2_rr )
  p1_avg = 0.5 * equations.gravity * h1_ll * h1_rr
  p2_avg = 0.5 * equations.gravity * h2_ll * h2_rr
  h1_v1_avg = 0.5 * (h1_v1_ll + h1_v1_rr )
  h1_w1_avg = 0.5 * (h1_w1_ll + h1_w1_rr )
  h2_v2_avg = 0.5 * (h2_v2_ll + h2_v2_rr )
  h2_w2_avg = 0.5 * (h2_w2_ll + h2_w2_rr )

  # Calculate fluxes depending on normal_direction
  f1 = h1_v1_avg * normal_direction[1] + h1_w1_avg * normal_direction[2]
  f2 = f1 * v1_avg + p1_avg * normal_direction[1]
  f3 = f1 * w1_avg + p1_avg * normal_direction[2]
  f4 = h2_v2_avg * normal_direction[1] + h2_w2_avg * normal_direction[2]
  f5 = f4 * v2_avg + p2_avg * normal_direction[1]
  f6 = f4 * w2_avg + p2_avg * normal_direction[2]

  return SVector(f1, f2, f3, f4, f5, f6, zero(eltype(u_ll)))
end


"""
    flux_es(u_ll, u_rr, orientation_or_normal_direction, equations::TwoLayerShallowWaterEquations1D)

Entropy stable surface flux for the two-layer shallow water equations. Uses the entropy stable 
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
@inline function flux_es(u_ll, u_rr,
                         orientation_or_normal_direction, 
                         equations::TwoLayerShallowWaterEquations2D)                   
  # Compute entropy conservative flux but without the bottom topography
  f_ec = flux_fjordholm_etal(u_ll, u_rr,
                            orientation_or_normal_direction,
                            equations)[1:6]

  # Get maximum signal velocity
  λ = max_abs_speed_naive(u_ll, u_rr, orientation_or_normal_direction, equations)

  # Get entropy variables but without the bottom topography
  q_rr = cons2entropy(u_rr,equations)[1:6]
  q_ll = cons2entropy(u_ll,equations)[1:6]

  # Average values from left and right 
  u_avg = (u_ll + u_rr)/2

  # Introduce variables for better readability
  rho1 = equations.rho1
  rho2 = equations.rho2
  g    = equations.gravity
  drho = rho1 - rho2

  # Entropy Jacobian matrix
  H = [[-rho2/(g*rho1*drho);;
        -rho2*u_avg[2]/(g*rho1*u_avg[1]*drho);;
        -rho2*u_avg[3]/(g*rho1*u_avg[1]*drho);;
        1.0/(g*drho);;
        u_avg[5]/(g*u_avg[4]*drho);;
        u_avg[6]/(g*u_avg[4]*drho)];
      [-rho2*u_avg[2]/(g*rho1*u_avg[1]*drho);;
        (g*rho1*u_avg[1]^3 - g*rho2*u_avg[1]^3 - rho2*u_avg[2]^2)/(g*rho1*u_avg[1]^2*drho);;
        -rho2*u_avg[2]*u_avg[3]/(g*rho1*u_avg[1]^2*drho);;
        u_avg[2]/(g*u_avg[1]*drho);;
        u_avg[2]*u_avg[5]/(g*u_avg[1]*u_avg[4]*drho);;
        u_avg[2]*u_avg[6]/(g*u_avg[1]*u_avg[4]*drho)];
      [-rho2*u_avg[3]/(g*rho1*u_avg[1]*drho);;
        -rho2*u_avg[2]*u_avg[3]/(g*rho1*u_avg[1]^2*drho);;
        (g*rho1*u_avg[1]^3 - g*rho2*u_avg[1]^3 - rho2*u_avg[3]^2)/(g*rho1*u_avg[1]^2*drho);;
        u_avg[3]/(g*u_avg[1]*drho);;
        u_avg[3]*u_avg[5]/(g*u_avg[1]*u_avg[4]*drho);;
        u_avg[3]*u_avg[6]/(g*u_avg[1]*u_avg[4]*drho)];
      [1.0/(g*drho);;
        u_avg[2]/(g*u_avg[1]*drho);;
        u_avg[3]/(g*u_avg[1]*drho);;
      -1.0/(g*drho);;
      -u_avg[5]/(g*u_avg[4]*drho);;
      -u_avg[6]/(g*u_avg[4]*drho)];
      [u_avg[5]/(g*u_avg[4]*drho);;
        u_avg[2]*u_avg[5]/(g*u_avg[1]*u_avg[4]*drho);;
        u_avg[3]*u_avg[5]/(g*u_avg[1]*u_avg[4]*drho);;
      -u_avg[5]/(g*u_avg[4]*drho);;
        (g*rho1*u_avg[4]^3 - g*rho2*u_avg[4]^3 - rho2*u_avg[5]^2)/(g*rho2*u_avg[4]^2*drho);;
      -u_avg[5]*u_avg[6]/(g*u_avg[4]^2*drho)];
      [u_avg[6]/(g*u_avg[4]*drho);;
        u_avg[2]*u_avg[6]/(g*u_avg[1]*u_avg[4]*drho);;
        u_avg[3]*u_avg[6]/(g*u_avg[1]*u_avg[4]*drho);;
      -u_avg[6]/(g*u_avg[4]*drho);;
      -u_avg[5]*u_avg[6]/(g*u_avg[4]^2*drho);;
        (g*rho1*u_avg[4]^3 - g*rho2*u_avg[4]^3 - rho2*u_avg[6]^2)/(g*rho2*u_avg[4]^2*drho)]]

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
@inline function max_abs_speed_naive(u_ll, u_rr, orientation::Integer, 
  equations::TwoLayerShallowWaterEquations2D)

  # Calculate averaged velocity of both layers
  if orientation == 1
    Um_ll = (u_ll[2] + u_ll[5]) / (u_ll[1] + u_ll[4])
    Um_rr = (u_rr[2] + u_rr[5]) / (u_rr[1] + u_rr[4])
  else
    Um_ll = (u_ll[3] + u_ll[6]) / (u_ll[1] + u_ll[4])
    Um_rr = (u_rr[3] + u_rr[6]) / (u_rr[1] + u_rr[4])
  end

  # Calculate the wave celerity on the left and right
  h1_ll, h2_ll = waterheight(u_ll, equations)
  h1_rr, h2_rr = waterheight(u_rr, equations)

  c_ll = sqrt(equations.gravity * (h1_ll + h2_ll) )
  c_rr = sqrt(equations.gravity * (h1_rr + h2_rr))

  return (max(abs(Um_ll),abs(Um_rr)) + max(c_ll,c_rr))
end


@inline function max_abs_speed_naive(u_ll, u_rr, 
                                     normal_direction::AbstractVector,
                                     equations::TwoLayerShallowWaterEquations2D)
  # Extract and compute the velocities in the normal direction
  v1_ll, w1_ll, v2_ll, w2_ll = velocity(u_ll, equations)
  v1_rr, w1_rr, v2_rr, w2_rr = velocity(u_rr, equations)

  v1_dot_n_ll = v1_ll * normal_direction[1] + w1_ll * normal_direction[2]
  v1_dot_n_rr = v1_rr * normal_direction[1] + w1_rr * normal_direction[2]
  v2_dot_n_ll = v2_ll * normal_direction[1] + w2_ll * normal_direction[2]
  v2_dot_n_rr = v2_rr * normal_direction[1] + w2_rr * normal_direction[2]
  
  # Calculate averaged velocity of both layers
  Um_ll = (v1_dot_n_ll * u_ll[1] + v2_dot_n_ll * u_ll[4]) / (u_ll[1] + u_ll[4])
  Um_rr = (v1_dot_n_rr * u_rr[1] + v2_dot_n_rr * u_rr[4]) / (u_rr[1] + u_rr[4])

  # Compute the wave celerity on the left and right
  h1_ll, h2_ll = waterheight(u_ll, equations)
  h1_rr, h2_rr = waterheight(u_rr, equations)

  c_ll = sqrt(equations.gravity * (h1_ll + h2_ll) )
  c_rr = sqrt(equations.gravity * (h1_rr + h2_rr))

  # The normal velocities are already scaled by the norm
  return max(abs(Um_ll), abs(Um_rr)) + max(c_ll, c_rr) * norm(normal_direction)
end


# Specialized `DissipationLocalLaxFriedrichs` to avoid spurious dissipation in the bottom topography
@inline function (dissipation::DissipationLocalLaxFriedrichs)(
      u_ll, u_rr, orientation_or_normal_direction, equations::TwoLayerShallowWaterEquations2D)
  λ = dissipation.max_abs_speed(u_ll, u_rr, orientation_or_normal_direction, equations)
  diss = -0.5 * λ * (u_rr - u_ll)
  return SVector(diss[1], diss[2], diss[3], diss[4], diss[5], diss[6], zero(eltype(u_ll)))
end


# Absolute speed of the barotropic mode
@inline function max_abs_speeds(u, equations::TwoLayerShallowWaterEquations2D)
  # Calculate averaged velocity of both layers
  v = (u[2] + u[5]) / (u[1] + u[4])
  w = (u[3] + u[6]) / (u[1] + u[4])

  h1, h2 = waterheight(u, equations)
  v1, w1, v2, w2 = velocity(u, equations)

  c = sqrt(equations.gravity * (h1 + h2)) 
  return max(abs(v) + c, abs(v1), abs(v2)), max(abs(w) + c, abs(w1), abs(w2))
end


# Helper function to extract the velocity vector from the conservative variables
@inline function velocity(u, equations::TwoLayerShallowWaterEquations2D)
  h1, h1_v1, h1_w1, h2, h2_v2, h2_w2, _ = u

  v1 = h1_v1 / h1
  w1 = h1_w1 / h1
  v2 = h2_v2 / h2
  w2 = h2_w2 / h2

  return v1, w1, v2, w2
end


# Convert conservative variables to primitive
@inline function cons2prim(u, equations::TwoLayerShallowWaterEquations2D)
  h1, h1_v1, h1_w1, h2, h2_v2, h2_w2, b = u

  H2 = h2 + b
  H1 = h2 + h1 + b
  v1, w1, v2, w2 = velocity(u, equations)

  return SVector(H1, v1, w1 , H2, v2, w2, b)
end


# Convert conservative variables to entropy variables
# Note, only the first four are the entropy variables, the fifth entry still just carries the bottom
# topography values for convenience. 
# In contrast to general usage the entropy variables are denoted with q instead of w, because w is
# already used for velocity in y-Direction
@inline function cons2entropy(u, equations::TwoLayerShallowWaterEquations2D)
  h1, h1_v1, h1_w1, h2, h2_v2, h2_w2, b = u
  # Assign new variables for better readability
  ρ1 = equations.rho1
  ρ2 = equations.rho2
  v1, w1, v2, w2 = velocity(u, equations)

  q1 = ρ1 * (equations.gravity * (              h1 + h2 + b) - 0.5 * (v1^2 + w1^2))
  q2 = ρ1 * v1
  q3 = ρ1 * w1
  q4 = ρ2 * (equations.gravity * (equations.r * h1 + h2 + b) - 0.5 * (v2^2 + w2^2))
  q5 = ρ2 * v2
  q6 = ρ2 * w2
  return SVector(q1, q2, q3, q4, q5, q6, b)
end


# Convert primitive to conservative variables
@inline function prim2cons(prim, equations::TwoLayerShallowWaterEquations2D)
  H1, v1, w1, H2, v2, w2, b = prim

  h2 = H2 - b
  h1 = H1 - h2 - b
  h1_v1 = h1 * v1
  h1_w1 = h1 * w1
  h2_v2 = h2 * v2
  h2_w2 = h2 * w2
  return SVector(h1, h1_v1, h1_w1, h2, h2_v2, h2_w2, b)
end


@inline function waterheight(u, equations::TwoLayerShallowWaterEquations2D)
  return u[1], u[4]
end


# Entropy function for the shallow water equations is the total energy
@inline entropy(cons, equations::TwoLayerShallowWaterEquations2D) = energy_total(cons, equations)


# Calculate total energy for a conservative state `cons`
@inline function energy_total(cons, equations::TwoLayerShallowWaterEquations2D)
  h1, h1_v1, h1_w1, h2, h2_v2, h2_w2, b = cons
  g = equations.gravity
  ρ1= equations.rho1
  ρ2= equations.rho2

  e = (0.5 * ρ1 * (h1_v1^2 / h1 + h1_w1^2 / h1 + g * h1^2) +
       0.5 * ρ2 * (h2_v2^2 / h2 + h2_w2^2 / h2 + g * h2^2) + g*ρ2*h2*b + g*ρ1*h1*(h2 + b))
  return e
end


# Calculate kinetic energy for a conservative state `cons`
@inline function energy_kinetic(u, equations::TwoLayerShallowWaterEquations2D)
  h1, h1_v1, h1_w1, h2, h2_v2, h2_w2, _ = u

  return (0.5 * equations.rho1 * h1_v1^2 / h1 + 0.5 * equations.rho1 * h1_w1^2 / h1 +
          0.5 * equations.rho2 * h2_v2^2 / h2 + 0.5 * equations.rho2 * h2_w2^2 / h2)
end


# Calculate potential energy for a conservative state `cons`
@inline function energy_internal(cons, equations::TwoLayerShallowWaterEquations2D)
  return energy_total(cons, equations) - energy_kinetic(cons, equations)
end


# Calculate the error for the "lake-at-rest" test case where H = h1+h2+b should
# be a constant value over time
@inline function lake_at_rest_error(u, equations::TwoLayerShallowWaterEquations2D)
  h1, _, _, h2, _, _, b = u
  return abs(equations.H0 - (h1 + h2 + b))
end

end # @muladd
