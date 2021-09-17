# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin


@doc raw"""
    ShallowWaterEquations2D(gravity, H0)

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
end

# Allow for flexibilty to set the gravitaional constant within an elixir depending on the
# application where `gravity_constant=1.0` or `gravity_constant=9.81` are common values.
# The reference total water height H0 defaults to 0.0 but is used for the "lake-at-rest"
# well-balancedness test cases
function ShallowWaterEquations2D(gravity_constant; H0=0.0)
  ShallowWaterEquations2D(gravity_constant, H0)
end


have_nonconservative_terms(::ShallowWaterEquations2D) = Val(true)
varnames(::typeof(cons2cons), ::ShallowWaterEquations2D) = ("h", "h_v1", "h_v2", "b")
varnames(::typeof(cons2prim), ::ShallowWaterEquations2D) = ("H", "v1", "v2", "b") # total water height: H = h + b


# Set initial conditions at physical location `x` for time `t`
"""
    initial_condition_constant(x, t, equations::ShallowWaterEquations2D)

A constant initial condition to test free-stream preservation / well-balancedness.
"""
function initial_condition_well_balancedness(x, t, equations::ShallowWaterEquations2D)
  # Set the background values
  H = equations.H0
  v1 = 0.0
  v2 = 0.0
  # bottom topography taken from Pond.control in [HOHQMesh](https://github.com/trixi-framework/HOHQMesh)
  x1, x2 = x
  b = (  1.5 / exp( 0.5 * ((x1 - 1.0)^2 + (x2 - 1.0)^2) )
       + 0.75 / exp( 0.5 * ((x1 + 1.0)^2 + (x2 + 1.0)^2) ) )
  return prim2cons(SVector(H, v1, v2, b), equations)
end


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
function boundary_condition_slip_wall(u_inner, normal_direction::AbstractVector, x, t,
                                      surface_flux_function, equations::ShallowWaterEquations2D)
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


# Calculate 1D flux for a single point in the normal direction
# Note, this directional vector is not normalized and the bottom topography has no flux
@inline function flux(u, normal_direction::AbstractVector, equations::ShallowWaterEquations2D)
  h = u[1]
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
@inline function flux_nonconservative_wintermeyer_etal(u_ll, u_rr,
                                                       normal_direction_ll::AbstractVector,
                                                       normal_direction_average::AbstractVector,
                                                       equations::ShallowWaterEquations2D)
  # Pull the necessary left and right state information
  h_ll = u_ll[1]
  b_rr = u_rr[4]
  # Note this routine only uses the `normal_direction_average` and the average of the
  # bottom topography to get a quadratic split form DG gradient on curved elements
  return SVector(zero(eltype(u_ll)),
                 normal_direction_average[1] * equations.gravity * h_ll * b_rr,
                 normal_direction_average[2] * equations.gravity * h_ll * b_rr,
                 zero(eltype(u_ll)))
end


"""
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
that account for possible discontinuities in the bottom topography function

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
    flux_fjordholm_etal(u_ll, u_rr, normal_direction,
                        equations::ShallowWaterEquations2D)

Total energy conservative (mathematical entropy for shallow water equations). When the bottom topography
is nonzero this should only be used as a surface flux otherwise the scheme will not be well-balanced.
For well-balancedness in the volume flux use [`flux_wintermeyer_etal`](@ref).

Details are available in Eq. (4.1) in the paper:
- Ulrik S. Fjordholm, Siddhartha Mishr and Eitan Tadmor (2011)
  Well-balanced and energy stable schemes for the shallow water equations with discontinuous topography
  [DOI: 10.1016/j.jcp.2011.03.042](https://doi.org/10.1016/j.jcp.2011.03.042)
"""
@inline function flux_fjordholm_etal(u_ll, u_rr, normal_direction::AbstractVector, equations::ShallowWaterEquations2D)
  # Unpack left and right state
  h_ll = u_ll[1]
  v1_ll, v2_ll = velocity(u_ll, equations)
  h_rr = u_rr[1]
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
    flux_wintermeyer_etal(u_ll, u_rr, normal_direction,
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
# TODO: This doesn't really use the `orientation` - should it?
@inline function max_abs_speed_naive(u_ll, u_rr, orientation::Integer, equations::ShallowWaterEquations2D)
  h_ll = u_ll[1]
  v1_ll, v2_ll = velocity(u_ll, equations)
  h_rr = u_rr[1]
  v1_rr, v2_rr = velocity(u_rr, equations)

  # Calculate velocity magnitude and wave celerity on the left and right
  v_mag_ll = sqrt(v1_ll^2 + v2_ll^2)
  c_ll = sqrt(equations.gravity * h_ll)
  v_mag_rr = sqrt(v1_rr^2 + v2_rr^2)
  c_rr = sqrt(equations.gravity * h_rr)

  λ_max = max(v_mag_ll, v_mag_rr) + max(c_ll, c_rr)
end

@inline function max_abs_speed_naive(u_ll, u_rr, normal_direction::AbstractVector, equations::ShallowWaterEquations2D)
  return max_abs_speed_naive(u_ll, u_rr, 0, equations) * norm(normal_direction)
end


@inline function min_max_speed_naive(u_ll, u_rr, normal_direction::AbstractVector,
                                     equations::ShallowWaterEquations2D)
  h_ll = u_ll[1]
  v1_ll, v2_ll = velocity(u_ll, equations)
  h_rr = u_rr[1]
  v1_rr, v2_rr = velocity(u_rr, equations)

  v_normal_ll = v1_ll * normal_direction[1] + v2_ll * normal_direction[2]
  v_normal_rr = v1_rr * normal_direction[1] + v2_rr * normal_direction[2]

  norm_ = norm(normal_direction)
  # The v_normals are already scaled by the norm
  λ_min = v_normal_ll - sqrt(equations.gravity * h_ll) * norm_
  λ_max = v_normal_rr + sqrt(equations.gravity * h_rr) * norm_

  return λ_min, λ_max
end


@inline function max_abs_speeds(u, equations::ShallowWaterEquations2D)
  h = u[1]
  v1, v2 = velocity(u, equations)

  c = equations.gravity * sqrt(h)
  return abs(v1) + c, abs(v2) + c
end


# Helper function to extract the velocity vector from the conservative variables
@inline function velocity(u, equations::ShallowWaterEquations2D)
  _, v1, v2, _ = cons2prim(u, equations)
  return SVector(v1, v2)
end


# Convert conservative variables to primitive
@inline function cons2prim(u, equations::ShallowWaterEquations2D)
  h, h_v1, h_v2, b = u

  H = h + b
  v1 = h_v1 / h
  v2 = h_v2 / h
  return SVector(H, v1, v2, b)
end


# Convert conservative variables to entropy
# Note, only the first three are the entropy variables, the fourth entry still
# just carries the bottom topography values for convenience
@inline function cons2entropy(u, equations::ShallowWaterEquations2D)
  h, h_v1, h_v2, b = u

  v1 = h_v1 / h
  v2 = h_v2 / h
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
  h = u[1]
  p = 0.5 * equations.gravity * h^2
  return p
end


@inline function waterheight_pressure(u, equations::ShallowWaterEquations2D)
  h = u[1]
  h_times_p = 0.5 * equations.gravity * h^3
  return h_times_p
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
@inline function lake_at_rest_error(u, equations::ShallowWaterEquations2D)
  return abs(equations.H0 - (u[1] + u[4]))
end

end # @muladd
