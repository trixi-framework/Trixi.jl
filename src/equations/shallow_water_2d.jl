# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin


@doc raw"""
    ShallowWaterEquations2D(gravity)

TODO: put in documentation
The bottom topography is not time dependent but it is stored as a fourth variable for
convenience
"""
struct ShallowWaterEquations2D{RealT<:Real} <: AbstractShallowWaterEquations{2, 4}
  gravity::RealT # gravitational constant

  function ShallowWaterEquations2D(gravity_constant)
    new{typeof(gravity_constant)}(gravity_constant)
  end
end

#have_nonconservative_terms(::ShallowWaterEquations2D) = Val(true)
varnames(::typeof(cons2cons), ::ShallowWaterEquations2D) = ("h", "h_v1", "h_v2", "b")
varnames(::typeof(cons2prim), ::ShallowWaterEquations2D) = ("h", "v1", "v2", "b")

# TODO: need to make sure that the initial conditions are h = H - b !
# Set initial conditions at physical location `x` for time `t`
"""
    initial_condition_constant(x, t, equations::ShallowWaterEquations2D)

A constant initial condition to test free-stream preservation or well-balancedness.
"""
function initial_condition_constant(x, t, equations::ShallowWaterEquations2D)
  h = 2.1
  h_v1 = 0.1
  h_v2 = -0.2
  b = bottom_topography(x, equations)
  return SVector(h, h_v1, h_v2, b)
end

# TODO: this manufactured solution and source term need updated with the bottom topography
"""
    initial_condition_convergence_test(x, t, equations::ShallowWaterEquations2D)

A smooth initial condition used for convergence tests in combination with
[`source_terms_convergence_test`](@ref)
(and [`BoundaryConditionDirichlet(initial_condition_convergence_test)`](@ref) in non-periodic domains).
"""
function initial_condition_convergence_test(x, t, equations::ShallowWaterEquations2D)
  # domain must be of length 2π in each direction to use periodic boundary conditions
  c  = 8.0
  v1 = 0.5
  v2 = 1.5

  h = c + cos(x[1]) * sin(x[2]) * cos(t)
  h_v1 = h * v1
  h_v2 = h * v2
  b = bottom_topography(x, equations)
  return SVector(h, h_v1, h_v2, b)
end

# TODO: update once the bottom topography is nonzero
"""
    source_terms_convergence_test(u, x, t, equations::ShallowWaterEquations2D)

Source terms used for convergence tests in combination with
[`initial_condition_convergence_test`](@ref)
(and [`BoundaryConditionDirichlet(initial_condition_convergence_test)`](@ref) in non-periodic domains).
"""
@inline function source_terms_convergence_test(u, x, t, equations::ShallowWaterEquations2D)
  # Same settings as in `initial_condition`
  c  = 8.0
  v1 = 0.5
  v2 = 1.5

  x1, x2 = x
  sinX, cosX = sincos(x1)
  sinY, cosY = sincos(x2)
  sinT, cosT = sincos(t)

  H   = c + cosX * sinY * cosT
  H_t = -cosX * sinY * sinT
  H_x = -sinX * sinY * cosT
  H_y =  cosX * cosY * cosT

  du1 = H_t + v1 * H_x + v2 * H_y
  du2 = v1 * du1 + equations.gravity * H * H_x
  du3 = v2 * du1 + equations.gravity * H * H_y
  return SVector(du1, du2, du3, 0.0)
end


"""
    initial_condition_weak_blast_wave(x, t, equations::ShallowWaterEquations2D)

A weak blast wave useful for testing, e.g., entropy conservation
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
  h  = r > 0.5 ? 1.0 : 1.1691
  v1 = r > 0.5 ? 0.0 : 0.1882 * cos_phi
  v2 = r > 0.5 ? 0.0 : 0.1882 * sin_phi
  b  = bottom_topography(x, equations)
  return prim2cons(SVector(h, v1, v2, b), equations)
end


# Calculate 1D flux for a single point
# Note the bottom topography has no flux
@inline function flux(u, orientation::Integer, equations::ShallowWaterEquations2D)
  h, h_v1, h_v2, _ = u

  v1 = h_v1 / h
  v2 = h_v2 / h
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
  h, v1, v2, _ = cons2prim(u, equations)

  v_normal = v1 * normal_direction[1] + v2 * normal_direction[2]
  h_v_normal = h * v_normal
  p = 0.5 * equations.gravity * h^2

  f1 = h_v_normal
  f2 = h_v_normal * v1 + p * normal_direction[1]
  f3 = h_v_normal * v2 + p * normal_direction[2]
  return SVector(f1, f2, f3, zero(eltype(u)))
end


# """
#     flux_nonconservative_powell(u_ll, u_rr, orientation::Integer,
#                                 equations::ShallowWaterEquations2D)
#     flux_nonconservative_powell(u_ll, u_rr,
#                                 normal_direction_ll     ::AbstractVector,
#                                 normal_direction_average::AbstractVector,
#                                 equations::ShallowWaterEquations2D)

# Non-symmetric two-point flux discretizing the nonconservative (source) term of
# that contains the gradient of the bottom topography [`ShallowWaterEquations2D`](@ref).

# On curvilinear meshes, this nonconservative flux depends on both the
# contravariant vector (normal direction) at the current node and the averaged
# one. This is different from numerical fluxes used to discretize conservative
# terms.

# ## References
# - wintermeyer
# """
# @inline function flux_nonconservative_wintermeyer(u_ll, u_rr, orientation::Integer,
#                                                   equations::ShallowWaterEquations2D)
#   h_ll = u_ll[1]
#   b_rr = u_rr[4]

#   # nonconservative term: (0, h b_x, h b_y, 0)
#   if orientation == 1
#     f = SVector(0, equations.gravity * h_ll * b_rr, 0, 0)
#   else # orientation == 2
#     f = SVector(0, 0, equations.gravity * h_ll * b_rr, 0)
#   end

#   return f
# end

# @inline function flux_nonconservative_wintermeyer(u_ll, u_rr,
#                                                   normal_direction_ll::AbstractVector,
#                                                   normal_direction_average::AbstractVector,
#                                                   equations::ShallowWaterEquations2D)
#   h_ll = u_ll[1]
#   b_rr = u_rr[4]

#   # Note this routine only uses the `normal_direction_ll` (contravariant vector
#   # at the same node location) with its components added together.
#   # The reason being that the terms on the left state multiplies some gradient.

#   # nonconservative term: (0, h b_x, h b_y, 0)
#   f = SVector(0,
#               v_dot_n_ll * psi_ll * psi_rr,
#               v_dot_n_ll * psi_rr)

#   return f
# end


"""
    flux_fjordholm_etal(u_ll, u_rr, orientation_or_normal_direction,
                        equations::ShallowWaterEquations2D)

Total energy conservative (mathematical entropy for shallow water equations). When the bottom topography
is nonzero this should only be used as a surface flux otherwise the scheme will not be well-balanced.
For well-balancedness in the volume flux use [`flux_wintermeyer_etal`](@ref).
- put in paper reference
"""
@inline function flux_fjordholm_etal(u_ll, u_rr, orientation::Integer, equations::ShallowWaterEquations2D)
  # Unpack left and right state
  h_ll, v1_ll, v2_ll, _ = cons2prim(u_ll, equations)
  h_rr, v1_rr, v2_rr, _ = cons2prim(u_rr, equations)

  # Average each factor of products in flux
  h_avg  = 0.5 * (h_ll   + h_rr  )
  v1_avg = 0.5 * (v1_ll  + v1_rr )
  v2_avg = 0.5 * (v2_ll  + v2_rr )
  h2_avg = 0.5 * (h_ll^2 + h_rr^2)
  p_avg  = 0.5 * equations.gravity * h2_avg

  # Calculate fluxes depending on orientation
  if orientation == 1
    f1 = h_avg * v1_avg
    f2 = h_avg * v1_avg * v1_avg + p_avg
    f3 = h_avg * v1_avg * v2_avg
  else
    f1 = h_avg * v2_avg
    f2 = h_avg * v2_avg * v1_avg
    f3 = h_avg * v2_avg * v2_avg + p_avg
  end

  return SVector(f1, f2, f3, zero(eltype(u_ll)))
end

@inline function flux_fjordholm_etal(u_ll, u_rr, normal_direction::AbstractVector, equations::ShallowWaterEquations2D)
  # Unpack left and right state
  h_ll, v1_ll, v2_ll, _ = cons2prim(u_ll, equations)
  h_rr, v1_rr, v2_rr, _ = cons2prim(u_rr, equations)

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
The `surface_flux` should still use, e.g., [`flux_fjordholm_etal`](@ref) or [`flux_hll](@ref).
- put in paper reference
"""
@inline function flux_wintermeyer_etal(u_ll, u_rr, orientation::Integer, equations::ShallowWaterEquations2D)
  # Unpack left and right state
  _, h_v1_ll, h_v2_ll, _ = u_ll
  _, h_v1_rr, h_v2_rr, _ = u_rr

  # Get the primitive variables
  h_ll, v1_ll, v2_ll, _ = cons2prim(u_ll, equations)
  h_rr, v1_rr, v2_rr, _ = cons2prim(u_rr, equations)

  # Average each factor of products in flux
  v1_avg = 0.5 * (v1_ll + v1_rr )
  v2_avg = 0.5 * (v2_ll + v2_rr )
  p_avg  = 0.5 * equations.gravity * h_ll * h_rr

  # Calculate fluxes depending on orientation
  if orientation == 1
    f1 = 0.5 * (h_v1_ll + h_v1_rr) # h_v1_avg
    f2 = f1 * v1_avg + p_avg
    f3 = f1 * v2_avg
  else
    f1 = 0.5 * (h_v2_ll + h_v2_rr) # h_v2_avg
    f2 = f1 * v1_avg
    f3 = f1 * v2_avg + p_avg
  end

  return SVector(f1, f2, f3, zero(eltype(u_ll)))
end

@inline function flux_wintermeyer_etal(u_ll, u_rr, normal_direction::AbstractVector, equations::ShallowWaterEquations2D)
  # Unpack left and right state
  _, h_v1_ll, h_v2_ll, _ = u_ll
  _, h_v1_rr, h_v2_rr, _ = u_rr

  # Get the primitive variables
  h_ll, v1_ll, v2_ll = cons2prim(u_ll, equations)
  h_rr, v1_rr, v2_rr = cons2prim(u_rr, equations)

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
  h_ll, v1_ll, v2_ll, _ = cons2prim(u_ll, equations)
  h_rr, v1_rr, v2_rr, _ = cons2prim(u_rr, equations)

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


# Calculate minimum and maximum wave speeds for HLL-type fluxes
@inline function min_max_speed_naive(u_ll, u_rr, orientation::Integer,
                                     equations::ShallowWaterEquations2D)
  h_ll, v1_ll, v2_ll, _ = cons2prim(u_ll, equations)
  h_rr, v1_rr, v2_rr, _ = cons2prim(u_rr, equations)

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
  h_ll, v1_ll, v2_ll, _ = cons2prim(u_ll, equations)
  h_rr, v1_rr, v2_rr, _ = cons2prim(u_rr, equations)

  v_normal_ll = v1_ll * normal_direction[1] + v2_ll * normal_direction[2]
  v_normal_rr = v1_rr * normal_direction[1] + v2_rr * normal_direction[2]

  norm_ = norm(normal_direction)
  # The v_normals are already scaled by the norm
  λ_min = v_normal_ll - sqrt(equations.gravity * h_ll) * norm_
  λ_max = v_normal_rr + sqrt(equations.gravity * h_rr) * norm_

  return λ_min, λ_max
end


# Called inside `FluxRotated` in `numerical_fluxes.jl` so the direction
# has been normalized prior to this rotation of the state vector
@inline function rotate_to_x(u, normal_vector, equations::ShallowWaterEquations2D)
  # cos and sin of the angle between the x-axis and the normalized normal_vector are
  # the normalized vector's x and y coordinates respectively (see unit circle).
  c = normal_vector[1]
  s = normal_vector[2]

  # Apply the 2D rotation matrix with normal and tangent directions of the form
  # [ 1    0    0   0;
  #   0   n_1  n_2  0;
  #   0   t_1  t_2  0;
  #   0    0    0   1 ]
  # where t_1 = -n_2 and t_2 = n_1

  return SVector(u[1],
                  c * u[2] + s * u[3],
                 -s * u[2] + c * u[3],
                 u[4])
end


# Called inside `FluxRotated` in `numerical_fluxes.jl` so the direction
# has been normalized prior to this back-rotation of the state vector
@inline function rotate_from_x(u, normal_vector, equations::ShallowWaterEquations2D)
  # cos and sin of the angle between the x-axis and the normalized normal_vector are
  # the normalized vector's x and y coordinates respectively (see unit circle).
  c = normal_vector[1]
  s = normal_vector[2]

  # Apply the 2D back-rotation matrix with normal and tangent directions of the form
  # [ 1    0    0   0;
  #   0   n_1  t_1  0;
  #   0   n_2  t_2  0;
  #   0    0    0   1 ]
  # where t_1 = -n_2 and t_2 = n_1

  return SVector(u[1],
                 c * u[2] - s * u[3],
                 s * u[2] + c * u[3],
                 u[4])
end


@inline function max_abs_speeds(u, equations::ShallowWaterEquations2D)
  h, v1, v2, _ = cons2prim(u, equations)

  c = equations.gravity * sqrt(h)
  return abs(v1) + c, abs(v2) + c
end


# TODO: figure out where this function should "live". probably always in the elixir
#       because it is part of the particular test case. But then would this function live in
#       the ShallowWaterEquations2D struct??
@inline function bottom_topography(x, equations::ShallowWaterEquations2D)
  b = 0.0
  # x1, x2 = x
  # b = ( 1.50 / exp(0.5 * ((x1 - 1.0)^2 + (x2 - 1.0)^2))
  #     + 0.75 / exp(0.5 * ((x1 + 1.0)^2 + (x2 + 1.0)^2)) )
  return b
end

# Convert conservative variables to primitive
@inline function cons2prim(u, equations::ShallowWaterEquations2D)
  h, h_v1, h_v2, b = u

  v1 = h_v1 / h
  v2 = h_v2 / h
  return SVector(h, v1, v2, b)
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

  h    = (w1 + 0.5 * (w2^2 + w3^2)) / equations.gravity - b
  h_v1 = h * w2
  h_v2 = h * w3
  return SVector(h, h_v1, h_v2, b)
end


# Convert primitive to conservative variables
@inline function prim2cons(prim, equations::ShallowWaterEquations2D)
  h, v1, v2, b = prim

  h_v1 = h * v1
  h_v2 = h * v2
  return SVector(h, h_v1, h_v2, b)
end


@inline function density(u, equations::ShallowWaterEquations2D)
  h = u[1]
  return h
end


@inline function pressure(u, equations::ShallowWaterEquations2D)
  h = u[1]
  p = 0.5 * equations.gravity * h^2
  return p
end


@inline function density_pressure(u, equations::ShallowWaterEquations2D)
  h = u[1]
  h_times_p = 0.5 * equations.gravity * h^3
  return h_times_p
end


# Entropy function for the shallow water equations is the total energy
@inline entropy(cons, equations::ShallowWaterEquations2D) = energy_total(cons, equations)


# Calculate total energy for a conservative state `cons`
@inline function energy_total(cons, ::ShallowWaterEquations2D)
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


end # @muladd
