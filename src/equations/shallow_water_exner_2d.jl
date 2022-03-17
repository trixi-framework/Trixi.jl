# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin


@doc raw"""
    ShallowWaterExnerEquations2D(gravity, H0, xi, Ag)

TODO: put in docstring and references, something about Grass model here
"""
struct ShallowWaterExnerEquations2D{RealT<:Real} <: AbstractShallowWaterExnerEquations{2, 4}
  gravity::RealT # gravitational constant
  H0::RealT      # constant "lake-at-rest" total water height
  xi::RealT      # 1 / (1 - porosity) scaling factor on the sediment discharge
  Ag::RealT      # constant in [0,1]; strength of interaction between fluid and sediment
end

function ShallowWaterExnerEquations2D(; gravity_constant, H0=0.0, porosity=0.0, Ag)
  ShallowWaterExnerEquations2D(gravity_constant, H0, inv(1.0 - porosity), Ag)
end


have_nonconservative_terms(::ShallowWaterExnerEquations2D) = Val(true)
varnames(::typeof(cons2cons), ::ShallowWaterExnerEquations2D) = ("h", "h_v1", "h_v2", "b")
# Note, we use the total water height, H = h + b, as the first primitive variable for easier
# visualization and setting initial conditions
varnames(::typeof(cons2prim), ::ShallowWaterExnerEquations2D) = ("H", "v1", "v2", "b")

"""
    boundary_condition_slip_wall(u_inner, normal_direction, x, t, surface_flux_function,
                                 equations::ShallowWaterExnerEquations2D)

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
                                      surface_flux_function, equations::ShallowWaterExnerEquations2D)
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


# Calculate 1D flux for a single point
# Note, for now this assumes a Grass model for the sediment discharge
@inline function flux(u, orientation::Integer, equations::ShallowWaterExnerEquations2D)
  h, h_v1, h_v2, _ = u
  v1, v2 = velocity(u, equations)

  p = 0.5 * equations.gravity * h^2
  if orientation == 1
    f1 = h_v1
    f2 = h_v1 * v1 + p
    f3 = h_v1 * v2
    f4 = equations.xi * equations.Ag * v1 * (v1^2 + v2^2)
  else
    f1 = h_v2
    f2 = h_v2 * v1
    f3 = h_v2 * v2 + p
    f4 = equations.xi * equations.Ag * v2 * (v1^2 + v2^2)
  end
  return SVector(f1, f2, f3, f4)
end


# Calculate 1D flux for a single point in the normal direction
# Note, this directional vector is not normalized and the bottom topography has no flux
@inline function flux(u, normal_direction::AbstractVector, equations::ShallowWaterExnerEquations2D)
  h = waterheight(u, equations)
  v1, v2 = velocity(u, equations)

  v_normal = v1 * normal_direction[1] + v2 * normal_direction[2]
  h_v_normal = h * v_normal
  p = 0.5 * equations.gravity * h^2

  f1 = h_v_normal
  f2 = h_v_normal * v1 + p * normal_direction[1]
  f3 = h_v_normal * v2 + p * normal_direction[2]
  f4 = equations.xi * equations.Ag * v_normal * (v1^2 + v2^2)
  return SVector(f1, f2, f3, f4)
end


"""
    flux_nonconservative_wintermeyer_etal(u_ll, u_rr, orientation::Integer,
                                          equations::ShallowWaterExnerEquations2D)
    flux_nonconservative_wintermeyer_etal(u_ll, u_rr,
                                          normal_direction_ll     ::AbstractVector,
                                          normal_direction_average::AbstractVector,
                                          equations::ShallowWaterExnerEquations2D)

Non-symmetric two-point volume flux discretizing the nonconservative (source) term
that contains the gradient of the bottom topography [`ShallowWaterExnerEquations2D`](@ref).

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
                                                       equations::ShallowWaterExnerEquations2D)
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
                                                       equations::ShallowWaterExnerEquations2D)
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
                                        equations::ShallowWaterExnerEquations2D)
    flux_nonconservative_fjordholm_etal(u_ll, u_rr,
                                        normal_direction_ll     ::AbstractVector,
                                        normal_direction_average::AbstractVector,
                                        equations::ShallowWaterExnerEquations2D)

Non-symmetric two-point surface flux discretizing the nonconservative (source) term of
that contains the gradient of the bottom topography [`ShallowWaterExnerEquations2D`](@ref).

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
                                                     equations::ShallowWaterExnerEquations2D)
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
                                                     equations::ShallowWaterExnerEquations2D)
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
    flux_fjordholm_etal(u_ll, u_rr, orientation_or_normal_direction,
                        equations::ShallowWaterExnerEquations2D)

Total energy conservative (mathematical entropy for shallow water equations). When the bottom topography
is nonzero this should only be used as a surface flux otherwise the scheme will not be well-balanced.
For well-balancedness in the volume flux use [`flux_wintermeyer_etal`](@ref).

Details are available in Eq. (4.1) in the paper:
- Ulrik S. Fjordholm, Siddhartha Mishr and Eitan Tadmor (2011)
  Well-balanced and energy stable schemes for the shallow water equations with discontinuous topography
  [DOI: 10.1016/j.jcp.2011.03.042](https://doi.org/10.1016/j.jcp.2011.03.042)
"""
@inline function flux_fjordholm_etal(u_ll, u_rr, orientation::Integer, equations::ShallowWaterExnerEquations2D)
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
  # TODO: the splitting the f4 flux is ad hoc currently
  if orientation == 1
    f1 = h_avg * v1_avg
    f2 = f1 * v1_avg + p_avg
    f3 = f1 * v2_avg
#    f4 = equations.xi * equations.Ag * v1_avg * (v1_avg^2 + v2_avg^2)
    f4_ll = equations.xi * equations.Ag * v1_ll * (v1_ll^2 + v2_ll^2)
    f4_rr = equations.xi * equations.Ag * v1_rr * (v1_rr^2 + v2_rr^2)
    f4 = 0.5 * (f4_ll + f4_rr)
  else
    f1 = h_avg * v2_avg
    f2 = f1 * v1_avg
    f3 = f1 * v2_avg + p_avg
#    f4 = equations.xi * equations.Ag * v2_avg * (v1_avg^2 + v2_avg^2)
    f4_ll = equations.xi * equations.Ag * v2_ll * (v1_ll^2 + v2_ll^2)
    f4_rr = equations.xi * equations.Ag * v2_rr * (v1_rr^2 + v2_rr^2)
    f4 = 0.5 * (f4_ll + f4_rr)
  end

  return SVector(f1, f2, f3, f4)
end

@inline function flux_fjordholm_etal(u_ll, u_rr, normal_direction::AbstractVector, equations::ShallowWaterExnerEquations2D)
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
  # TODO: the splitting the f4 flux is ad hoc currently
#  f4 = equations.xi * equations.Ag * v_dot_n_avg * (v1_avg^2 + v2_avg^2)
  f4_ll = equations.xi * equations.Ag * v_dot_n_ll * (v1_ll^2 + v2_ll^2)
  f4_rr = equations.xi * equations.Ag * v_dot_n_rr * (v1_rr^2 + v2_rr^2)
  f4 = 0.5 * (f4_ll + f4_rr)

  return SVector(f1, f2, f3, f4)
end


"""
    flux_wintermeyer_etal(u_ll, u_rr, orientation_or_normal_direction,
                          equations::ShallowWaterExnerEquations2D)

Total energy conservative (mathematical entropy for shallow water equations) split form.
When the bottom topography is nonzero this scheme will be well-balanced when used as a `volume_flux`.
The `surface_flux` should still use, e.g., [`flux_fjordholm_etal`](@ref).

Further details are available in Theorem 1 of the paper:
- Niklas Wintermeyer, Andrew R. Winters, Gregor J. Gassner and David A. Kopriva (2017)
  An entropy stable nodal discontinuous Galerkin method for the two dimensional
  shallow water equations on unstructured curvilinear meshes with discontinuous bathymetry
  [DOI: 10.1016/j.jcp.2017.03.036](https://doi.org/10.1016/j.jcp.2017.03.036)
"""
@inline function flux_wintermeyer_etal(u_ll, u_rr, orientation::Integer, equations::ShallowWaterExnerEquations2D)
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
  # TODO: the splitting the f4 flux is ad hoc currently
  if orientation == 1
    f1 = 0.5 * (h_v1_ll + h_v1_rr)
    f2 = f1 * v1_avg + p_avg
    f3 = f1 * v2_avg
#    f4 = equations.xi * equations.Ag * v1_avg * (v1_avg^2 + v2_avg^2)
    f4_ll = equations.xi * equations.Ag * v1_ll * (v1_ll^2 + v2_ll^2)
    f4_rr = equations.xi * equations.Ag * v1_rr * (v1_rr^2 + v2_rr^2)
    f4 = 0.5 * (f4_ll + f4_rr)
  else
    f1 = 0.5 * (h_v2_ll + h_v2_rr)
    f2 = f1 * v1_avg
    f3 = f1 * v2_avg + p_avg
#    f4 = equations.xi * equations.Ag * v2_avg * (v1_avg^2 + v2_avg^2)
    f4_ll = equations.xi * equations.Ag * v2_ll * (v1_ll^2 + v2_ll^2)
    f4_rr = equations.xi * equations.Ag * v2_rr * (v1_rr^2 + v2_rr^2)
    f4 = 0.5 * (f4_ll + f4_rr)
  end

  return SVector(f1, f2, f3, f4)
end

@inline function flux_wintermeyer_etal(u_ll, u_rr, normal_direction::AbstractVector, equations::ShallowWaterExnerEquations2D)
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
  # TODO: the splitting the f4 flux is ad hoc currently
  v_dot_n_ll = v1_ll * normal_direction[1] + v2_ll * normal_direction[2]
  v_dot_n_rr = v1_rr * normal_direction[1] + v2_rr * normal_direction[2]
  # v_dot_n_avg = 0.5 * (v_dot_n_ll + v_dot_n_rr)
  # f4 = equations.xi * equations.Ag * v_dot_n_avg * (v1_avg^2 + v2_avg^2)
  f4_ll = equations.xi * equations.Ag * v_dot_n_ll * (v1_ll^2 + v2_ll^2)
  f4_rr = equations.xi * equations.Ag * v_dot_n_rr * (v1_rr^2 + v2_rr^2)
  f4 = 0.5 * (f4_ll + f4_rr)

  return SVector(f1, f2, f3, f4)
end


# Calculate maximum wave speed for local Lax-Friedrichs-type dissipation as the
# maximum velocity magnitude plus the maximum speed of sound
@inline function max_abs_speed_naive(u_ll, u_rr, orientation::Integer, equations::ShallowWaterExnerEquations2D)
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

@inline function max_abs_speed_naive(u_ll, u_rr, normal_direction::AbstractVector, equations::ShallowWaterExnerEquations2D)
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


# Calculate minimum and maximum wave speeds for HLL-type fluxes
@inline function min_max_speed_naive(u_ll, u_rr, orientation::Integer,
                                     equations::ShallowWaterExnerEquations2D)
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
                                     equations::ShallowWaterExnerEquations2D)
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


@inline function max_abs_speeds(u, equations::ShallowWaterExnerEquations2D)
  h = waterheight(u, equations)
  v1, v2 = velocity(u, equations)

  c = equations.gravity * sqrt(h)
  return abs(v1) + c, abs(v2) + c
end


# Helper function to extract the velocity vector from the conservative variables
@inline function velocity(u, equations::ShallowWaterExnerEquations2D)
  h, h_v1, h_v2, _ = u

  v1 = h_v1 / h
  v2 = h_v2 / h
  return SVector(v1, v2)
end


# Convert conservative variables to primitive
@inline function cons2prim(u, equations::ShallowWaterExnerEquations2D)
  h, _, _, b = u

  H = h + b
  v1, v2 = velocity(u, equations)
  return SVector(H, v1, v2, b)
end


# Convert conservative variables to entropy
# Note, only the first three are the entropy variables, the fourth entry still
# just carries the bottom topography values for convenience
@inline function cons2entropy(u, equations::ShallowWaterExnerEquations2D)
  h, _, _, b = u

  v1, v2 = velocity(u, equations)
  v_square = v1^2 + v2^2

  w1 = equations.gravity * (h + b) - 0.5 * v_square
  w2 = v1
  w3 = v2
  w4 = equations.gravity * h
  return SVector(w1, w2, w3, w4)
end


# Convert entropy variables to conservative
@inline function entropy2cons(w, equations::ShallowWaterExnerEquations2D)
  w1, w2, w3, b = w

  h = (w1 + 0.5 * (w2^2 + w3^2)) / equations.gravity - b
  h_v1 = h * w2
  h_v2 = h * w3
  return SVector(h, h_v1, h_v2, b)
end


# Convert primitive to conservative variables
@inline function prim2cons(prim, equations::ShallowWaterExnerEquations2D)
  H, v1, v2, b = prim

  h = H - b
  h_v1 = h * v1
  h_v2 = h * v2
  return SVector(h, h_v1, h_v2, b)
end


@inline function waterheight(u, equations::ShallowWaterExnerEquations2D)
  return u[1]
end


# Entropy function for the shallow water equations is the total energy
@inline entropy(cons, equations::ShallowWaterExnerEquations2D) = energy_total(cons, equations)


# Calculate total energy for a conservative state `cons`
@inline function energy_total(cons, equations::ShallowWaterExnerEquations2D)
  h, h_v1, h_v2, b = cons

  e = (h_v1^2 + h_v2^2) / (2 * h) + 0.5 * equations.gravity * h^2 + equations.gravity * h * b
  return e
end


# Calculate kinetic energy for a conservative state `cons`
@inline function energy_kinetic(u, equations::ShallowWaterExnerEquations2D)
  h, h_v1, h_v2, _ = u
  return (h_v1^2 + h_v2^2) / (2 * h)
end


# Calculate potential energy for a conservative state `cons`
@inline function energy_internal(cons, equations::ShallowWaterExnerEquations2D)
  return energy_total(cons, equations) - energy_kinetic(cons, equations)
end


# Calculate the error for the "lake-at-rest" test case where H = h+b should
# be a constant value over time
@inline function lake_at_rest_error(u, equations::ShallowWaterExnerEquations2D)
  h, _, _, b = u
  return abs(equations.H0 - (h + b))
end

end # @muladd
