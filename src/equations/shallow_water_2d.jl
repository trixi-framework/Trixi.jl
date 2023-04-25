# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin


@doc raw"""
    ShallowWaterEquations2D(gravity, H0, threshold_limiter, threshold_wet)

Shallow water equations (SWE) in two space dimensions. The equations are given by
```math
\begin{aligned}
  \frac{\partial h}{\partial t} + \frac{\partial}{\partial x}(h v_1)
    + \frac{\partial}{\partial y}(h v_2) &= 0 \\
    \frac{\partial}{\partial t}(h v_1) + \frac{\partial}{\partial x}\left(h v_1^2 + \frac{g}{2}h^2\right)
    + \frac{\partial}{\partial y}(h v_1 v_2) + g h \frac{\partial b}{\partial x} &= 0 \\
    \frac{\partial}{\partial t}(h v_2) + \frac{\partial}{\partial x}(h v_1 v_2)
    + \frac{\partial}{\partial y}\left(h v_2^2 + \frac{g}{2}h^2\right) + g h \frac{\partial b}{\partial y} &= 0.
\end{aligned}
```
The unknown quantities of the SWE are the water height ``h`` and the velocities ``\mathbf{v} = (v_1, v_2)^T``.
The gravitational constant is denoted by `g` and the (possibly) variable bottom topography function ``b(x,y)``.
Conservative variable water height ``h`` is measured from the bottom topography ``b``, therefore one
also defines the total water height as ``H = h + b``.

The additional quantity ``H_0`` is also available to store a reference value for the total water height that
is useful to set initial conditions or test the "lake-at-rest" well-balancedness.

Also, there are two thresholds which prevent numerical problems as well as instabilities. Both of them do not
have to be passed, as default values are defined within the struct. The first one, `threshold_limiter`, is
used in [`PositivityPreservingLimiterShallowWater`](@ref) on the water height, as a (small) shift on the initial
condition and cutoff before the next time step. The second one, `threshold_wet`, is applied on the water height to
define when the flow is "wet" before calculating the numerical flux.

The bottom topography function ``b(x,y)`` is set inside the initial condition routine
for a particular problem setup. To test the conservative form of the SWE one can set the bottom topography
variable `b` to zero.

In addition to the unknowns, Trixi currently stores the bottom topography values at the approximation points
despite being fixed in time. This is done for convenience of computing the bottom topography gradients
on the fly during the approximation as well as computing auxiliary quantities like the total water height ``H``
or the entropy variables.
This affects the implementation and use of these equations in various ways:
* The flux values corresponding to the bottom topography must be zero.
* The bottom topography values must be included when defining initial conditions, boundary conditions or
  source terms.
* [`AnalysisCallback`](@ref) analyzes this variable.
* Trixi's visualization tools will visualize the bottom topography by default.

References for the SWE are many but a good introduction is available in Chapter 13 of the book:
- Randall J. LeVeque (2002)
  Finite Volume Methods for Hyperbolic Problems
  [DOI: 10.1017/CBO9780511791253](https://doi.org/10.1017/CBO9780511791253)
"""
struct ShallowWaterEquations2D{RealT<:Real} <: AbstractShallowWaterEquations{2, 4}
  gravity::RealT # gravitational constant
  H0::RealT      # constant "lake-at-rest" total water height
  threshold_limiter::RealT  # Threshold to use in PositivityPreservingLimiterShallowWater on water height,
                             # as a (small) shift on the initial condition and cutoff before the 
                             # next time step.
   threshold_wet::RealT      # Threshold to be applied on water height to define when the flow is "wet"
                             # before calculating the numerical flux.
end

# Allow for flexibility to set the gravitational constant within an elixir depending on the
# application where `gravity_constant=1.0` or `gravity_constant=9.81` are common values.
# The reference total water height H0 is an artefact from the old calculation of the lake_at_rest_error
# Strict default values for thresholds that performed great in several numerical experiments
function ShallowWaterEquations2D(; gravity_constant, H0=0.0,
                                 threshold_limiter=1e-13, threshold_wet=1e-15)
  ShallowWaterEquations2D(gravity_constant, H0, threshold_limiter, threshold_wet)
end


have_nonconservative_terms(::ShallowWaterEquations2D) = True()
varnames(::typeof(cons2cons), ::ShallowWaterEquations2D) = ("h", "h_v1", "h_v2", "b")
# Note, we use the total water height, H = h + b, as the first primitive variable for easier
# visualization and setting initial conditions
varnames(::typeof(cons2prim), ::ShallowWaterEquations2D) = ("H", "v1", "v2", "b")


# Set initial conditions at physical location `x` for time `t`
"""
    initial_condition_convergence_test(x, t, equations::ShallowWaterEquations2D)

A smooth initial condition used for convergence tests in combination with
[`source_terms_convergence_test`](@ref)
(and [`BoundaryConditionDirichlet(initial_condition_convergence_test)`](@ref) in non-periodic domains).
"""
function initial_condition_convergence_test(x, t, equations::ShallowWaterEquations2D)
  # some constants are chosen such that the function is periodic on the domain [0,sqrt(2)]^2
  c  = 7.0
  omega_x = 2.0 * pi * sqrt(2.0)
  omega_t = 2.0 * pi

  x1, x2 = x

  H = c + cos(omega_x * x1) * sin(omega_x * x2) * cos(omega_t * t)
  v1 = 0.5
  v2 = 1.5
  b = 2.0 + 0.5 * sin(sqrt(2.0) * pi * x1) + 0.5 * sin(sqrt(2.0) * pi * x2)
  return prim2cons(SVector(H, v1, v2, b), equations)
end

"""
    source_terms_convergence_test(u, x, t, equations::ShallowWaterEquations2D)

Source terms used for convergence tests in combination with
[`initial_condition_convergence_test`](@ref)
(and [`BoundaryConditionDirichlet(initial_condition_convergence_test)`](@ref) in non-periodic domains).

This manufactured solution source term is specifically designed for the bottom topography function
`b(x,y) = 2 + 0.5 * sin(sqrt(2)*pi*x) + 0.5 * sin(sqrt(2)*pi*y)`
as defined in [`initial_condition_convergence_test`](@ref).
"""
@inline function source_terms_convergence_test(u, x, t, equations::ShallowWaterEquations2D)
  # Same settings as in `initial_condition_convergence_test`. Some derivative simplify because
  # this manufactured solution velocities are taken to be constants
  c  = 7.0
  omega_x = 2.0 * pi * sqrt(2.0)
  omega_t = 2.0 * pi
  omega_b = sqrt(2.0) * pi
  v1 = 0.5
  v2 = 1.5

  x1, x2 = x

  sinX, cosX = sincos(omega_x * x1)
  sinY, cosY = sincos(omega_x * x2)
  sinT, cosT = sincos(omega_t * t )

  H = c + cosX * sinY * cosT
  H_x = -omega_x * sinX * sinY * cosT
  H_y =  omega_x * cosX * cosY * cosT
  # this time derivative for the water height exploits that the bottom topography is
  # fixed in time such that H_t = (h+b)_t = h_t + 0
  H_t = -omega_t * cosX * sinY * sinT

  # bottom topography and its gradient
  b = 2.0 + 0.5 * sin(sqrt(2.0) * pi * x1) + 0.5 * sin(sqrt(2.0) * pi * x2)
  tmp1 = 0.5 * omega_b
  b_x = tmp1 * cos(omega_b * x1)
  b_y = tmp1 * cos(omega_b * x2)

  du1 = H_t + v1 * (H_x - b_x) + v2 * (H_y - b_y)
  du2 = v1 * du1 + equations.gravity * (H - b) * H_x
  du3 = v2 * du1 + equations.gravity * (H - b) * H_y
  return SVector(du1, du2, du3, 0.0)
end


"""
    initial_condition_weak_blast_wave(x, t, equations::ShallowWaterEquations2D)

A weak blast wave discontinuity useful for testing, e.g., total energy conservation.
Note for the shallow water equations to the total energy acts as a mathematical entropy function.
"""
function initial_condition_weak_blast_wave(x, t, equations::ShallowWaterEquations2D)
  # Set up polar coordinates
  inicenter = SVector(0.7, 0.7)
  x_norm = x[1] - inicenter[1]
  y_norm = x[2] - inicenter[2]
  r = sqrt(x_norm^2 + y_norm^2)
  phi = atan(y_norm, x_norm)
  sin_phi, cos_phi = sincos(phi)

  # Calculate primitive variables
  H = r > 0.5 ? 3.25 : 4.0
  v1 = r > 0.5 ? 0.0 : 0.1882 * cos_phi
  v2 = r > 0.5 ? 0.0 : 0.1882 * sin_phi
  b = 0.0 # by default assume there is no bottom topography

  return prim2cons(SVector(H, v1, v2, b), equations)
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
                                              x, t,
                                              surface_flux_function,
                                              equations::ShallowWaterEquations2D)
  # normalize the outward pointing direction
  normal = normal_direction / norm(normal_direction)

  # compute the normal velocity
  u_normal = normal[1] * u_inner[2] + normal[2] * u_inner[3]

  # create the "external" boundary solution state
  u_boundary = SVector(u_inner[1],
                       u_inner[2] - 2.0 * u_normal * normal[1],
                       u_inner[3] - 2.0 * u_normal * normal[2],
                       u_inner[4])

  # calculate the boundary flux
  flux = surface_flux_function(u_inner, u_boundary, normal_direction, equations)

  return flux
end


"""
    boundary_condition_slip_wall(u_inner, orientation, direction, x, t,
                                 surface_flux_function, equations::ShallowWaterEquations2D)

Should be used together with [`TreeMesh`](@ref).
"""
@inline function boundary_condition_slip_wall(u_inner, orientation,
                                              direction, x, t,
                                              surface_flux_function,
                                              equations::ShallowWaterEquations2D)
  ## get the appropriate normal vector from the orientation
  if orientation == 1
    u_boundary = SVector(u_inner[1], -u_inner[2], u_inner[3], u_inner[4])
  else # orientation == 2
    u_boundary = SVector(u_inner[1], u_inner[2], -u_inner[3], u_inner[4])
  end

  # compute and return the flux using `boundary_condition_slip_wall` routine above
  flux = surface_flux_function(u_inner, u_boundary, orientation, equations)

  return flux
end

# Calculate 1D flux for a single point
# Note, the bottom topography has no flux
@inline function flux(u, orientation::Integer, equations::ShallowWaterEquations2D)
  h, h_v1, h_v2, _ = u
  v1, v2 = velocity(u, equations)

  p = 0.5 * equations.gravity * h^2
  if orientation == 1
    f1 = h_v1
    f2 = h_v1 * v1 + p
    f3 = h_v1 * v2
  else
    f1 = h_v2
    f2 = h_v2 * v1
    f3 = h_v2 * v2 + p
  end
  return SVector(f1, f2, f3, zero(eltype(u)))
end


# Calculate 1D flux for a single point in the normal direction
# Note, this directional vector is not normalized and the bottom topography has no flux
@inline function flux(u, normal_direction::AbstractVector, equations::ShallowWaterEquations2D)
  h = waterheight(u, equations)
  v1, v2 = velocity(u, equations)

  v_normal = v1 * normal_direction[1] + v2 * normal_direction[2]
  h_v_normal = h * v_normal
  p = 0.5 * equations.gravity * h^2

  f1 = h_v_normal
  f2 = h_v_normal * v1 + p * normal_direction[1]
  f3 = h_v_normal * v2 + p * normal_direction[2]
  return SVector(f1, f2, f3, zero(eltype(u)))
end


"""
    flux_nonconservative_wintermeyer_etal(u_ll, u_rr, orientation::Integer,
                                          equations::ShallowWaterEquations2D)
    flux_nonconservative_wintermeyer_etal(u_ll, u_rr,
                                          normal_direction_ll     ::AbstractVector,
                                          normal_direction_average::AbstractVector,
                                          equations::ShallowWaterEquations2D)

Non-symmetric two-point volume flux discretizing the nonconservative (source) term
that contains the gradient of the bottom topography [`ShallowWaterEquations2D`](@ref).

On curvilinear meshes, this nonconservative flux depends on both the
contravariant vector (normal direction) at the current node and the averaged
one. This is different from numerical fluxes used to discretize conservative
terms.

Further details are available in the paper:
- Niklas Wintermeyer, Andrew R. Winters, Gregor J. Gassner and David A. Kopriva (2017)
  An entropy stable nodal discontinuous Galerkin method for the two dimensional
  shallow water equations on unstructured curvilinear meshes with discontinuous bathymetry
  [DOI: 10.1016/j.jcp.2017.03.036](https://doi.org/10.1016/j.jcp.2017.03.036)
"""
@inline function flux_nonconservative_wintermeyer_etal(u_ll, u_rr, orientation::Integer,
                                                       equations::ShallowWaterEquations2D)
  # Pull the necessary left and right state information
  h_ll = waterheight(u_ll, equations)
  b_rr = u_rr[4]

  z = zero(eltype(u_ll))
  # Bottom gradient nonconservative term: (0, g h b_x, g h b_y, 0)
  if orientation == 1
    f = SVector(z, equations.gravity * h_ll * b_rr, z, z)
  else # orientation == 2
    f = SVector(z, z, equations.gravity * h_ll * b_rr, z)
  end
  return f
end

@inline function flux_nonconservative_wintermeyer_etal(u_ll, u_rr,
                                                       normal_direction_ll::AbstractVector,
                                                       normal_direction_average::AbstractVector,
                                                       equations::ShallowWaterEquations2D)
  # Pull the necessary left and right state information
  h_ll = waterheight(u_ll, equations)
  b_rr = u_rr[4]
  # Note this routine only uses the `normal_direction_average` and the average of the
  # bottom topography to get a quadratic split form DG gradient on curved elements
  return SVector(zero(eltype(u_ll)),
                 normal_direction_average[1] * equations.gravity * h_ll * b_rr,
                 normal_direction_average[2] * equations.gravity * h_ll * b_rr,
                 zero(eltype(u_ll)))
end


"""
    flux_nonconservative_fjordholm_etal(u_ll, u_rr, orientation::Integer,
                                        equations::ShallowWaterEquations2D)
    flux_nonconservative_fjordholm_etal(u_ll, u_rr,
                                        normal_direction_ll     ::AbstractVector,
                                        normal_direction_average::AbstractVector,
                                        equations::ShallowWaterEquations2D)

Non-symmetric two-point surface flux discretizing the nonconservative (source) term of
that contains the gradient of the bottom topography [`ShallowWaterEquations2D`](@ref).

On curvilinear meshes, this nonconservative flux depends on both the
contravariant vector (normal direction) at the current node and the averaged
one. This is different from numerical fluxes used to discretize conservative
terms.

This contains additional terms compared to [`flux_nonconservative_wintermeyer_etal`](@ref)
that account for possible discontinuities in the bottom topography function.
Thus, this flux should be used in general at interfaces. For flux differencing volume terms,
[`flux_nonconservative_wintermeyer_etal`](@ref) is analytically equivalent but slightly
cheaper.

Further details for the original finite volume formulation are available in
- Ulrik S. Fjordholm, Siddhartha Mishr and Eitan Tadmor (2011)
  Well-balanced and energy stable schemes for the shallow water equations with discontinuous topography
  [DOI: 10.1016/j.jcp.2011.03.042](https://doi.org/10.1016/j.jcp.2011.03.042)
and for curvilinear 2D case in the paper:
- Niklas Wintermeyer, Andrew R. Winters, Gregor J. Gassner and David A. Kopriva (2017)
  An entropy stable nodal discontinuous Galerkin method for the two dimensional
  shallow water equations on unstructured curvilinear meshes with discontinuous bathymetry
  [DOI: 10.1016/j.jcp.2017.03.036](https://doi.org/10.1016/j.jcp.2017.03.036)
"""
@inline function flux_nonconservative_fjordholm_etal(u_ll, u_rr, orientation::Integer,
                                                     equations::ShallowWaterEquations2D)
  # Pull the necessary left and right state information
  h_ll, _, _, b_ll = u_ll
  h_rr, _, _, b_rr = u_rr

  h_average = 0.5 * (h_ll + h_rr)
  b_jump = b_rr - b_ll

  # Includes two parts:
  #   (i)  Diagonal (consistent) term from the volume flux that uses `b_ll` to avoid
  #        cross-averaging across a discontinuous bottom topography
  #   (ii) True surface part that uses `h_average` and `b_jump` to handle discontinuous bathymetry
  z = zero(eltype(u_ll))
  if orientation == 1
    f = SVector(z,
                equations.gravity * h_ll * b_ll + equations.gravity * h_average * b_jump,
                z, z)
  else # orientation == 2
    f = SVector(z, z,
                equations.gravity * h_ll * b_ll + equations.gravity * h_average * b_jump,
                z)
  end

  return f
end

@inline function flux_nonconservative_fjordholm_etal(u_ll, u_rr,
                                                     normal_direction_ll::AbstractVector,
                                                     normal_direction_average::AbstractVector,
                                                     equations::ShallowWaterEquations2D)
  # Pull the necessary left and right state information
  h_ll, _, _, b_ll = u_ll
  h_rr, _, _, b_rr = u_rr

  # Comes in two parts:
  #   (i)  Diagonal (consistent) term from the volume flux that uses `normal_direction_average`
  #        but we use `b_ll` to avoid cross-averaging across a discontinuous bottom topography

  f2 = normal_direction_average[1] * equations.gravity * h_ll * b_ll
  f3 = normal_direction_average[2] * equations.gravity * h_ll * b_ll

  #   (ii) True surface part that uses `normal_direction_ll`, `h_average` and `b_jump`
  #        to handle discontinuous bathymetry
  h_average = 0.5 * (h_ll + h_rr)
  b_jump = b_rr - b_ll

  f2 += normal_direction_ll[1] * equations.gravity * h_average * b_jump
  f3 += normal_direction_ll[2] * equations.gravity * h_average * b_jump

  # First and last equations do not have a nonconservative flux
  f1 = f4 = zero(eltype(u_ll))

  return SVector(f1, f2, f3, f4)
end


"""
    hydrostatic_reconstruction_audusse_etal(u_ll, u_rr, orientation_or_normal_direction,
                                            equations::ShallowWaterEquations2D)

A particular type of hydrostatic reconstruction on the water height to guarantee well-balancedness
for a general bottom topography [`ShallowWaterEquations2D`](@ref). The reconstructed solution states
`u_ll_star` and `u_rr_star` variables are used to evaluate the surface numerical flux at the interface.
Use in combination with the generic numerical flux routine [`FluxHydrostaticReconstruction`](@ref).

Further details for the hydrostatic reconstruction and its motivation can be found in
- Emmanuel Audusse, François Bouchut, Marie-Odile Bristeau, Rupert Klein, and Benoit Perthame (2004)
  A fast and stable well-balanced scheme with hydrostatic reconstruction for shallow water flows
  [DOI: 10.1137/S1064827503431090](https://doi.org/10.1137/S1064827503431090)
"""
@inline function hydrostatic_reconstruction_audusse_etal(u_ll, u_rr, equations::ShallowWaterEquations2D)
  # Unpack left and right water heights and bottom topographies
  h_ll, _, _, b_ll = u_ll
  h_rr, _, _, b_rr = u_rr

  # Get the velocities on either side
  v1_ll, v2_ll = velocity(u_ll, equations)
  v1_rr, v2_rr = velocity(u_rr, equations)

  # Compute the reconstructed water heights
  h_ll_star = max(zero(h_ll) , h_ll + b_ll - max(b_ll, b_rr) )
  h_rr_star = max(zero(h_rr) , h_rr + b_rr - max(b_ll, b_rr) )

  # Create the conservative variables using the reconstruted water heights
  u_ll_star = SVector( h_ll_star , h_ll_star * v1_ll , h_ll_star * v2_ll , b_ll )
  u_rr_star = SVector( h_rr_star , h_rr_star * v1_rr , h_rr_star * v2_rr , b_rr )

  return u_ll_star, u_rr_star
end

"""
    hydrostatic_reconstruction_chen_noelle(u_ll, u_rr, orientation::Integer,
                                           equations::ShallowWaterEquations2D)

A particular type of hydrostatic reconstruction on the water height to guarantee well-balancedness
for a general bottom topography [`ShallowWaterEquations2D`](@ref). The reconstructed solution states
`u_ll_star` and `u_rr_star` variables are used to evaluate the surface numerical flux at the interface.
The key idea is a linear reconstruction of the bottom and water height at the interfaces using subcells.
Use in combination with the generic numerical flux routine [`FluxHydrostaticReconstruction`](@ref).

Further details on this hydrostatic reconstruction and its motivation can be found in
- Guoxian Chen and Sebastian Noelle (2017) 
  A new hydrostatic reconstruction scheme based on subcell reconstructions
  [DOI:10.1137/15M1053074](https://dx.doi.org/10.1137/15M1053074)
"""
@inline function hydrostatic_reconstruction_chen_noelle(u_ll, u_rr, equations::ShallowWaterEquations2D)
  # Unpack left and right water heights and bottom topographies
  h_ll, _, _, b_ll = u_ll
  h_rr, _, _, b_rr = u_rr

  # Get the velocities on either side
  v1_ll, v2_ll = velocity(u_ll, equations)
  v1_rr, v2_rr = velocity(u_rr, equations)

  H_ll = b_ll + h_ll
  H_rr = b_rr + h_rr

  b_star = min( max( b_ll, b_rr ), min( H_ll, H_rr ) )
  
  # Compute the reconstructed water heights
  h_ll_star = min( H_ll - b_star, h_ll )
  h_rr_star = min( H_rr - b_star, h_rr )

  # Set the water height to be at least the value stored in the variable threshold after
  # the hydrostatic reconstruction is applied and before the numerical flux is calculated
  # to avoid numerical problem with arbitrary small values. Interfaces with a water height
  # lower or equal to the threshold can be declared as dry.
  # The default value is set to 1e-15 and can be changed within the constructor call in an elixir.
  threshold = equations.threshold_wet

  h_ll_star = h_ll_star * Int32(h_ll_star > threshold) + threshold * Int32(h_ll_star <= threshold)
  h_rr_star = h_rr_star * Int32(h_rr_star > threshold) + threshold * Int32(h_rr_star <= threshold)

  v1_ll = v1_ll * Int32(h_ll_star > threshold)
  v1_rr = v1_rr * Int32(h_rr_star > threshold)

  v2_ll = v2_ll * Int32(h_ll_star > threshold)
  v2_rr = v2_rr * Int32(h_rr_star > threshold)

  # Create the conservative variables using the reconstruted water heights
  u_ll_star = SVector( h_ll_star, h_ll_star * v1_ll, h_ll_star * v2_ll, b_ll )
  u_rr_star = SVector( h_rr_star, h_rr_star * v1_rr, h_rr_star * v2_rr, b_rr )

  return u_ll_star, u_rr_star
end

"""
    flux_nonconservative_audusse_etal(u_ll, u_rr, orientation::Integer,
                                      equations::ShallowWaterEquations2D)
    flux_nonconservative_audusse_etal(u_ll, u_rr,
                                      normal_direction_ll     ::AbstractVector,
                                      normal_direction_average::AbstractVector,
                                      equations::ShallowWaterEquations2D)

Non-symmetric two-point surface flux that discretizes the nonconservative (source) term.
The discretization uses the `hydrostatic_reconstruction_audusse_etal` on the conservative
variables.

This hydrostatic reconstruction ensures that the finite volume numerical fluxes remain
well-balanced for discontinuous bottom topographies [`ShallowWaterEquations2D`](@ref).
Should be used together with [`FluxHydrostaticReconstruction`](@ref) and
[`hydrostatic_reconstruction_audusse_etal`](@ref) in the surface flux to ensure consistency.

Further details for the hydrostatic reconstruction and its motivation can be found in
- Emmanuel Audusse, François Bouchut, Marie-Odile Bristeau, Rupert Klein, and Benoit Perthame (2004)
  A fast and stable well-balanced scheme with hydrostatic reconstruction for shallow water flows
  [DOI: 10.1137/S1064827503431090](https://doi.org/10.1137/S1064827503431090)
"""
@inline function flux_nonconservative_audusse_etal(u_ll, u_rr, orientation::Integer,
                                                   equations::ShallowWaterEquations2D)
  # Pull the water height and bottom topography on the left
  h_ll, _, _, b_ll = u_ll

  # Create the hydrostatic reconstruction for the left solution state
  u_ll_star, _ = hydrostatic_reconstruction_audusse_etal(u_ll, u_rr, equations)

  # Copy the reconstructed water height for easier to read code
  h_ll_star = u_ll_star[1]

  z = zero(eltype(u_ll))
  # Includes two parts:
  #   (i)  Diagonal (consistent) term from the volume flux that uses `b_ll` to avoid
  #        cross-averaging across a discontinuous bottom topography
  #   (ii) True surface part that uses `h_ll` and `h_ll_star` to handle discontinuous bathymetry
  if orientation == 1
    f = SVector(z,
                equations.gravity * h_ll * b_ll + equations.gravity * ( h_ll^2 - h_ll_star^2 ),
                z, z)
  else # orientation == 2
    f = SVector(z, z,
                equations.gravity * h_ll * b_ll + equations.gravity * ( h_ll^2 - h_ll_star^2 ),
                z)
  end

  return f
end

@inline function flux_nonconservative_audusse_etal(u_ll, u_rr,
                                                   normal_direction_ll::AbstractVector,
                                                   normal_direction_average::AbstractVector,
                                                   equations::ShallowWaterEquations2D)
  # Pull the water height and bottom topography on the left
  h_ll, _, _, b_ll = u_ll

  # Create the hydrostatic reconstruction for the left solution state
  u_ll_star, _ = hydrostatic_reconstruction_audusse_etal(u_ll, u_rr, equations)

  # Copy the reconstructed water height for easier to read code
  h_ll_star = u_ll_star[1]

  # Comes in two parts:
  #   (i)  Diagonal (consistent) term from the volume flux that uses `normal_direction_average`
  #        but we use `b_ll` to avoid cross-averaging across a discontinuous bottom topography

  f2 = normal_direction_average[1] * equations.gravity * h_ll * b_ll
  f3 = normal_direction_average[2] * equations.gravity * h_ll * b_ll

  #   (ii) True surface part that uses `normal_direction_ll`, `h_ll` and `h_ll_star`
  #        to handle discontinuous bathymetry

  f2 += normal_direction_ll[1] * equations.gravity * ( h_ll^2 - h_ll_star^2 )
  f3 += normal_direction_ll[2] * equations.gravity * ( h_ll^2 - h_ll_star^2 )

  # First and last equations do not have a nonconservative flux
  f1 = f4 = zero(eltype(u_ll))

  return SVector(f1, f2, f3, f4)
end

"""
    flux_nonconservative_chen_noelle(u_ll, u_rr,
                                     orientation::Integer,
                                     equations::ShallowWaterEquations2D)
    flux_nonconservative_chen_noelle(u_ll, u_rr,
                                     normal_direction_ll      ::AbstractVector,
                                     normal_direction_average ::AbstractVector,
                                     equations::ShallowWaterEquations2D)

Non-symmetric two-point surface flux that discretizes the nonconservative (source) term.
The discretization uses the [`hydrostatic_reconstruction_chen_noelle`](@ref) on the conservative
variables.

Should be used together with [`FluxHydrostaticReconstruction`](@ref) and
[`hydrostatic_reconstruction_chen_noelle`](@ref) in the surface flux to ensure consistency.

Further details on the hydrostatic reconstruction and its motivation can be found in
- Guoxian Chen and Sebastian Noelle (2017) 
  A new hydrostatic reconstruction scheme based on subcell reconstructions
  [DOI:10.1137/15M1053074](https://dx.doi.org/10.1137/15M1053074)
"""
@inline function flux_nonconservative_chen_noelle(u_ll, u_rr, orientation::Integer,
                                                  equations::ShallowWaterEquations2D)
  # Pull the water height and bottom topography on the left
  h_ll, _, _, b_ll = u_ll
  h_rr, _, _, b_rr = u_rr

  H_ll = h_ll + b_ll
  H_rr = h_rr + b_rr

  b_star = min( max( b_ll, b_rr ), min( H_ll, H_rr ) )

  # Create the hydrostatic reconstruction for the left solution state
  u_ll_star, _ = hydrostatic_reconstruction_chen_noelle(u_ll, u_rr, equations)

  # Copy the reconstructed water height for easier to read code
  h_ll_star = u_ll_star[1]

  z = zero(eltype(u_ll))
  # Includes two parts:
  #   (i)  Diagonal (consistent) term from the volume flux that uses `b_ll` to avoid
  #        cross-averaging across a discontinuous bottom topography
  #   (ii) True surface part that uses `h_ll` and `h_ll_star` to handle discontinuous bathymetry
  if orientation == 1
    f = SVector(z,
                equations.gravity * h_ll * b_ll - equations.gravity * (h_ll_star + h_ll) * (b_ll - b_star),
                z, z)
  else # orientation == 2
    f = SVector(z, z,
                equations.gravity * h_ll * b_ll - equations.gravity * (h_ll_star + h_ll) * (b_ll - b_star),
                z)
  end

  return f
end

@inline function flux_nonconservative_chen_noelle(u_ll, u_rr,
                                                   normal_direction_ll::AbstractVector,
                                                   normal_direction_average::AbstractVector,
                                                   equations::ShallowWaterEquations2D)
  # Pull the water height and bottom topography on the left
  h_ll, _, _, b_ll = u_ll
  h_rr, _, _, b_rr = u_rr

  H_ll = h_ll + b_ll
  H_rr = h_rr + b_rr

  b_star = min( max( b_ll, b_rr ), min( H_ll, H_rr ) )
  
  # Create the hydrostatic reconstruction for the left solution state
  u_ll_star, _ = hydrostatic_reconstruction_chen_noelle(u_ll, u_rr, equations)

  # Copy the reconstructed water height for easier to read code
  h_ll_star = u_ll_star[1]

  # Comes in two parts:
  #   (i)  Diagonal (consistent) term from the volume flux that uses `normal_direction_average`
  #        but we use `b_ll` to avoid cross-averaging across a discontinuous bottom topography

  f2 = normal_direction_average[1] * equations.gravity * h_ll * b_ll
  f3 = normal_direction_average[2] * equations.gravity * h_ll * b_ll

  #   (ii) True surface part that uses `normal_direction_ll`, `h_ll` and `h_ll_star`
  #        to handle discontinuous bathymetry

  f2 -= normal_direction_ll[1] * equations.gravity * (h_ll_star + h_ll) * (b_ll - b_star)
  f3 -= normal_direction_ll[2] * equations.gravity * (h_ll_star + h_ll) * (b_ll - b_star)

  # First and last equations do not have a nonconservative flux
  f1 = f4 = zero(eltype(u_ll))

  return SVector(f1, f2, f3, f4)
end


"""
    flux_fjordholm_etal(u_ll, u_rr, orientation_or_normal_direction,
                        equations::ShallowWaterEquations2D)

Total energy conservative (mathematical entropy for shallow water equations). When the bottom topography
is nonzero this should only be used as a surface flux otherwise the scheme will not be well-balanced.
For well-balancedness in the volume flux use [`flux_wintermeyer_etal`](@ref).

Details are available in Eq. (4.1) in the paper:
- Ulrik S. Fjordholm, Siddhartha Mishr and Eitan Tadmor (2011)
  Well-balanced and energy stable schemes for the shallow water equations with discontinuous topography
  [DOI: 10.1016/j.jcp.2011.03.042](https://doi.org/10.1016/j.jcp.2011.03.042)
"""
@inline function flux_fjordholm_etal(u_ll, u_rr, orientation::Integer, equations::ShallowWaterEquations2D)
  # Unpack left and right state
  h_ll = waterheight(u_ll, equations)
  v1_ll, v2_ll = velocity(u_ll, equations)
  h_rr = waterheight(u_rr, equations)
  v1_rr, v2_rr = velocity(u_rr, equations)

  # Average each factor of products in flux
  h_avg  = 0.5 * (h_ll   + h_rr  )
  v1_avg = 0.5 * (v1_ll  + v1_rr )
  v2_avg = 0.5 * (v2_ll  + v2_rr )
  p_avg  = 0.25 * equations.gravity * (h_ll^2 + h_rr^2)

  # Calculate fluxes depending on orientation
  if orientation == 1
    f1 = h_avg * v1_avg
    f2 = f1 * v1_avg + p_avg
    f3 = f1 * v2_avg
  else
    f1 = h_avg * v2_avg
    f2 = f1 * v1_avg
    f3 = f1 * v2_avg + p_avg
  end

  return SVector(f1, f2, f3, zero(eltype(u_ll)))
end

@inline function flux_fjordholm_etal(u_ll, u_rr, normal_direction::AbstractVector, equations::ShallowWaterEquations2D)
  # Unpack left and right state
  h_ll = waterheight(u_ll, equations)
  v1_ll, v2_ll = velocity(u_ll, equations)
  h_rr = waterheight(u_rr, equations)
  v1_rr, v2_rr = velocity(u_rr, equations)

  v_dot_n_ll = v1_ll * normal_direction[1] + v2_ll * normal_direction[2]
  v_dot_n_rr = v1_rr * normal_direction[1] + v2_rr * normal_direction[2]

  # Average each factor of products in flux
  h_avg  = 0.5 * (h_ll   + h_rr  )
  v1_avg = 0.5 * (v1_ll  + v1_rr )
  v2_avg = 0.5 * (v2_ll  + v2_rr )
  h2_avg = 0.5 * (h_ll^2 + h_rr^2)
  p_avg  = 0.5 * equations.gravity * h2_avg
  v_dot_n_avg = 0.5 * (v_dot_n_ll + v_dot_n_rr)

  # Calculate fluxes depending on normal_direction
  f1 = h_avg * v_dot_n_avg
  f2 = f1 * v1_avg + p_avg * normal_direction[1]
  f3 = f1 * v2_avg + p_avg * normal_direction[2]

  return SVector(f1, f2, f3, zero(eltype(u_ll)))
end


"""
    flux_wintermeyer_etal(u_ll, u_rr, orientation_or_normal_direction,
                          equations::ShallowWaterEquations2D)

Total energy conservative (mathematical entropy for shallow water equations) split form.
When the bottom topography is nonzero this scheme will be well-balanced when used as a `volume_flux`.
The `surface_flux` should still use, e.g., [`flux_fjordholm_etal`](@ref).

Further details are available in Theorem 1 of the paper:
- Niklas Wintermeyer, Andrew R. Winters, Gregor J. Gassner and David A. Kopriva (2017)
  An entropy stable nodal discontinuous Galerkin method for the two dimensional
  shallow water equations on unstructured curvilinear meshes with discontinuous bathymetry
  [DOI: 10.1016/j.jcp.2017.03.036](https://doi.org/10.1016/j.jcp.2017.03.036)
"""
@inline function flux_wintermeyer_etal(u_ll, u_rr, orientation::Integer, equations::ShallowWaterEquations2D)
  # Unpack left and right state
  h_ll, h_v1_ll, h_v2_ll, _ = u_ll
  h_rr, h_v1_rr, h_v2_rr, _ = u_rr

  # Get the velocities on either side
  v1_ll, v2_ll = velocity(u_ll, equations)
  v1_rr, v2_rr = velocity(u_rr, equations)

  # Average each factor of products in flux
  v1_avg = 0.5 * (v1_ll + v1_rr)
  v2_avg = 0.5 * (v2_ll + v2_rr)
  p_avg  = 0.5 * equations.gravity * h_ll * h_rr

  # Calculate fluxes depending on orientation
  if orientation == 1
    f1 = 0.5 * (h_v1_ll + h_v1_rr)
    f2 = f1 * v1_avg + p_avg
    f3 = f1 * v2_avg
  else
    f1 = 0.5 * (h_v2_ll + h_v2_rr)
    f2 = f1 * v1_avg
    f3 = f1 * v2_avg + p_avg
  end

  return SVector(f1, f2, f3, zero(eltype(u_ll)))
end

@inline function flux_wintermeyer_etal(u_ll, u_rr, normal_direction::AbstractVector, equations::ShallowWaterEquations2D)
  # Unpack left and right state
  h_ll, h_v1_ll, h_v2_ll, _ = u_ll
  h_rr, h_v1_rr, h_v2_rr, _ = u_rr

  # Get the velocities on either side
  v1_ll, v2_ll = velocity(u_ll, equations)
  v1_rr, v2_rr = velocity(u_rr, equations)

  # Average each factor of products in flux
  h_v1_avg = 0.5 * (h_v1_ll + h_v1_rr )
  h_v2_avg = 0.5 * (h_v2_ll + h_v2_rr )
  v1_avg   = 0.5 * (v1_ll   + v1_rr   )
  v2_avg   = 0.5 * (v2_ll   + v2_rr   )
  p_avg    = 0.5 * equations.gravity * h_ll * h_rr

  # Calculate fluxes depending on normal_direction
  f1 = h_v1_avg * normal_direction[1] + h_v2_avg * normal_direction[2]
  f2 = f1 * v1_avg + p_avg * normal_direction[1]
  f3 = f1 * v2_avg + p_avg * normal_direction[2]

  return SVector(f1, f2, f3, zero(eltype(u_ll)))
end


# Calculate maximum wave speed for local Lax-Friedrichs-type dissipation as the
# maximum velocity magnitude plus the maximum speed of sound
@inline function max_abs_speed_naive(u_ll, u_rr, orientation::Integer, equations::ShallowWaterEquations2D)
  # Get the velocity quantities in the appropriate direction
  if orientation == 1
    v_ll, _ = velocity(u_ll, equations)
    v_rr, _ = velocity(u_rr, equations)
  else
    _, v_ll = velocity(u_ll, equations)
    _, v_rr = velocity(u_rr, equations)
  end

  # Calculate the wave celerity on the left and right
  h_ll = waterheight(u_ll, equations)
  h_rr = waterheight(u_rr, equations)
  c_ll = sqrt(equations.gravity * h_ll)
  c_rr = sqrt(equations.gravity * h_rr)

  return max(abs(v_ll), abs(v_rr)) + max(c_ll, c_rr)
end

@inline function max_abs_speed_naive(u_ll, u_rr, normal_direction::AbstractVector, equations::ShallowWaterEquations2D)
  # Extract and compute the velocities in the normal direction
  v1_ll, v2_ll = velocity(u_ll, equations)
  v1_rr, v2_rr = velocity(u_rr, equations)
  v_ll = v1_ll * normal_direction[1] + v2_ll * normal_direction[2]
  v_rr = v1_rr * normal_direction[1] + v2_rr * normal_direction[2]

  # Compute the wave celerity on the left and right
  h_ll = waterheight(u_ll, equations)
  h_rr = waterheight(u_rr, equations)
  c_ll = sqrt(equations.gravity * h_ll)
  c_rr = sqrt(equations.gravity * h_rr)

  # The normal velocities are already scaled by the norm
  return max(abs(v_ll), abs(v_rr)) + max(c_ll, c_rr) * norm(normal_direction)
end


# Specialized `DissipationLocalLaxFriedrichs` to avoid spurious dissipation in the bottom topography
@inline function (dissipation::DissipationLocalLaxFriedrichs)(u_ll, u_rr, orientation_or_normal_direction,
                                                              equations::ShallowWaterEquations2D)
  λ = dissipation.max_abs_speed(u_ll, u_rr, orientation_or_normal_direction, equations)
  diss = -0.5 * λ * (u_rr - u_ll)
  return SVector(diss[1], diss[2], diss[3], zero(eltype(u_ll)))
end


# Specialized `FluxHLL` to avoid spurious dissipation in the bottom topography
@inline function (numflux::FluxHLL)(u_ll, u_rr, orientation_or_normal_direction,
                  equations::ShallowWaterEquations2D)
  λ_min, λ_max = numflux.min_max_speed(u_ll, u_rr, orientation_or_normal_direction, equations)

  if λ_min >= 0 && λ_max >= 0
    return flux(u_ll, orientation_or_normal_direction, equations)
  elseif λ_max <= 0 && λ_min <= 0
    return flux(u_rr, orientation_or_normal_direction, equations)
  else
    f_ll = flux(u_ll, orientation_or_normal_direction, equations)
    f_rr = flux(u_rr, orientation_or_normal_direction, equations)
    inv_λ_max_minus_λ_min = inv(λ_max - λ_min)
    factor_ll = λ_max * inv_λ_max_minus_λ_min
    factor_rr = λ_min * inv_λ_max_minus_λ_min
    factor_diss = λ_min * λ_max * inv_λ_max_minus_λ_min
    diss = u_rr - u_ll
    return factor_ll * f_ll - factor_rr * f_rr + factor_diss * SVector(diss[1], diss[2], diss[3], zero(eltype(u_ll)))
  end
end


# Calculate minimum and maximum wave speeds for HLL-type fluxes
@inline function min_max_speed_naive(u_ll, u_rr, orientation::Integer,
                                     equations::ShallowWaterEquations2D)
  h_ll = waterheight(u_ll, equations)
  v1_ll, v2_ll = velocity(u_ll, equations)
  h_rr = waterheight(u_rr, equations)
  v1_rr, v2_rr = velocity(u_rr, equations)

  if orientation == 1 # x-direction
    λ_min = v1_ll - sqrt(equations.gravity * h_ll)
    λ_max = v1_rr + sqrt(equations.gravity * h_rr)
  else # y-direction
    λ_min = v2_ll - sqrt(equations.gravity * h_ll)
    λ_max = v2_rr + sqrt(equations.gravity * h_rr)
  end

  return λ_min, λ_max
end

@inline function min_max_speed_naive(u_ll, u_rr, normal_direction::AbstractVector,
                                     equations::ShallowWaterEquations2D)
  h_ll = waterheight(u_ll, equations)
  v1_ll, v2_ll = velocity(u_ll, equations)
  h_rr = waterheight(u_rr, equations)
  v1_rr, v2_rr = velocity(u_rr, equations)

  v_normal_ll = v1_ll * normal_direction[1] + v2_ll * normal_direction[2]
  v_normal_rr = v1_rr * normal_direction[1] + v2_rr * normal_direction[2]

  norm_ = norm(normal_direction)
  # The v_normals are already scaled by the norm
  λ_min = v_normal_ll - sqrt(equations.gravity * h_ll) * norm_
  λ_max = v_normal_rr + sqrt(equations.gravity * h_rr) * norm_

  return λ_min, λ_max
end

"""
    min_max_speed_chen_noelle(u_ll, u_rr, orientation::Integer,
                              equations::ShallowWaterEquations2D)
    min_max_speed_chen_noelle(u_ll, u_rr, normal_direction::AbstractVector,
                              equations::ShallowWaterEquations2D)

The approximated speeds for the HLL type numerical flux used by Chen and Noelle for their 
hydrostatic reconstruction. As they state in the paper, those speeds are chosen for the numerical
flux to ensure positivity and satisfy an entropy inequality.

Further details on this hydrostatic reconstruction and its motivation can be found in
- Guoxian Chen and Sebastian Noelle (2017) 
  A new hydrostatic reconstruction scheme based on subcell reconstructions
  [DOI:10.1137/15M1053074](https://dx.doi.org/10.1137/15M1053074)
"""
@inline function min_max_speed_chen_noelle(u_ll, u_rr, orientation::Integer, 
                                           equations::ShallowWaterEquations2D)
  h_ll = waterheight(u_ll, equations)
  v1_ll, v2_ll = velocity(u_ll, equations)
  h_rr = waterheight(u_rr, equations)
  v1_rr, v2_rr = velocity(u_rr, equations)

  a_ll = sqrt(equations.gravity * h_ll)
  a_rr = sqrt(equations.gravity * h_rr)

  if orientation == 1 # x-direction
    λ_min = min( v1_ll - a_ll, v1_rr - a_rr, zero(eltype(u_ll)) ) 
    λ_max = max( v1_ll + a_ll, v1_rr + a_rr, zero(eltype(u_ll)) )
  else # y-direction
    λ_min = min( v2_ll - a_ll, v2_rr - a_rr, zero(eltype(u_ll)) )
    λ_max = max( v2_ll + a_ll, v2_rr + a_rr, zero(eltype(u_ll)) )
  end
  return λ_min, λ_max
end

@inline function min_max_speed_chen_noelle(u_ll, u_rr, normal_direction::AbstractVector,
                                           equations::ShallowWaterEquations2D)
  h_ll = waterheight(u_ll, equations)
  v1_ll, v2_ll = velocity(u_ll, equations)
  h_rr = waterheight(u_rr, equations)
  v1_rr, v2_rr = velocity(u_rr, equations)

  v_normal_ll = v1_ll * normal_direction[1] + v2_ll * normal_direction[2]
  v_normal_rr = v1_rr * normal_direction[1] + v2_rr * normal_direction[2]

  norm_ = norm(normal_direction)

  a_ll = sqrt(equations.gravity * h_ll) * norm_
  a_rr = sqrt(equations.gravity * h_rr) * norm_

  λ_min = min( v_normal_ll - a_ll, v_normal_rr - a_rr, zero(eltype(u_ll)) ) 
  λ_max = max( v_normal_ll + a_ll, v_normal_rr + a_rr, zero(eltype(u_ll)) )

  return λ_min, λ_max
end

@inline function max_abs_speeds(u, equations::ShallowWaterEquations2D)
  h = waterheight(u, equations)
  v1, v2 = velocity(u, equations)

  c = equations.gravity * sqrt(h)
  return abs(v1) + c, abs(v2) + c
end


# Helper function to extract the velocity vector from the conservative variables
@inline function velocity(u, equations::ShallowWaterEquations2D)
  h, h_v1, h_v2, _ = u

  v1 = h_v1 / h
  v2 = h_v2 / h
  return SVector(v1, v2)
end


# Convert conservative variables to primitive
@inline function cons2prim(u, equations::ShallowWaterEquations2D)
  h, _, _, b = u

  H = h + b
  v1, v2 = velocity(u, equations)
  return SVector(H, v1, v2, b)
end


# Convert conservative variables to entropy
# Note, only the first three are the entropy variables, the fourth entry still
# just carries the bottom topography values for convenience
@inline function cons2entropy(u, equations::ShallowWaterEquations2D)
  h, h_v1, h_v2, b = u

  v1, v2 = velocity(u, equations)
  v_square = v1^2 + v2^2

  w1 = equations.gravity * (h + b) - 0.5 * v_square
  w2 = v1
  w3 = v2
  return SVector(w1, w2, w3, b)
end


# Convert entropy variables to conservative
@inline function entropy2cons(w, equations::ShallowWaterEquations2D)
  w1, w2, w3, b = w

  h = (w1 + 0.5 * (w2^2 + w3^2)) / equations.gravity - b
  h_v1 = h * w2
  h_v2 = h * w3
  return SVector(h, h_v1, h_v2, b)
end


# Convert primitive to conservative variables
@inline function prim2cons(prim, equations::ShallowWaterEquations2D)
  H, v1, v2, b = prim

  h = H - b
  h_v1 = h * v1
  h_v2 = h * v2
  return SVector(h, h_v1, h_v2, b)
end


@inline function waterheight(u, equations::ShallowWaterEquations2D)
  return u[1]
end


@inline function pressure(u, equations::ShallowWaterEquations2D)
  h = waterheight(u, equations)
  p = 0.5 * equations.gravity * h^2
  return p
end


@inline function waterheight_pressure(u, equations::ShallowWaterEquations2D)
  return waterheight(u, equations) * pressure(u, equations)
end


# Entropy function for the shallow water equations is the total energy
@inline entropy(cons, equations::ShallowWaterEquations2D) = energy_total(cons, equations)


# Calculate total energy for a conservative state `cons`
@inline function energy_total(cons, equations::ShallowWaterEquations2D)
  h, h_v1, h_v2, b = cons

  e = (h_v1^2 + h_v2^2) / (2 * h) + 0.5 * equations.gravity * h^2 + equations.gravity * h * b
  return e
end


# Calculate kinetic energy for a conservative state `cons`
@inline function energy_kinetic(u, equations::ShallowWaterEquations2D)
  h, h_v1, h_v2, _ = u
  return (h_v1^2 + h_v2^2) / (2 * h)
end


# Calculate potential energy for a conservative state `cons`
@inline function energy_internal(cons, equations::ShallowWaterEquations2D)
  return energy_total(cons, equations) - energy_kinetic(cons, equations)
end


# Calculate the error for the "lake-at-rest" test case where H = h+b should
# be a constant value over time
@inline function lake_at_rest_error(u, u_exact, equations::ShallowWaterEquations2D)
  h, _, _, b = u
  h_exact, _, _, b_exact= u_exact

  H = h + b
  H_exact = h_exact + b_exact

  return abs(H - H_exact)
end

end # @muladd
