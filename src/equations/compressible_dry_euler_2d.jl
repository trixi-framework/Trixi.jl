# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin


struct CompressibleDryEulerEquations2D{RealT<:Real} <: AbstractCompressibleDryEulerEquations{2, 4}
  p_0::RealT   # constant reference pressure 1000 hPa(100000 Pa)
  c_p::RealT
  c_v::RealT
  R_d::RealT   # gas constant
  g::RealT # gravitation constant
  kappa::RealT # ratio of the gas constand R_d
  gamma::RealT # = inv(kappa- 1); can be used to write slow divisions as fast multiplications
  a::RealT
end

function CompressibleDryEulerEquations2D(;RealT=Float64)
   p_0 = 100000.0
   c_p = 1004.0
   c_v = 717.0
   R_d = c_p-c_v 
   g = 9.81
   gamma = c_p / c_v # = 1/(1 - kappa)
   kappa = 1 - inv(gamma)
   a=360
   return CompressibleDryEulerEquations2D{RealT}(p_0, c_p, c_v, R_d, g, kappa, gamma, a)
  end


varnames(::typeof(cons2cons), ::CompressibleDryEulerEquations2D) = ("rho", "rho_v1", "rho_v2", "rho_e")
varnames(::typeof(cons2prim), ::CompressibleDryEulerEquations2D) = ("rho", "v1", "v2", "p")
varnames(::typeof(cons2pot), ::CompressibleDryEulerEquations2D) = ("rho", "v1", "v2", "pottemp")


function boundary_condition_slip_wall(u_inner, normal_direction::AbstractVector, x, t,
                                      surface_flux_function, equations::CompressibleDryEulerEquations2D)

  norm_ = norm(normal_direction)
  # Normalize the vector without using `normalize` since we need to multiply by the `norm_` later
  normal = normal_direction / norm_

  # rotate the internal solution state
  u_local = rotate_to_x(u_inner, normal, equations)

  # compute the primitive variables
  rho_local, v_normal, v_tangent, p_local = cons2prim(u_local, equations)

  # Get the solution of the pressure Riemann problem
  # See Section 6.3.3 of
  # Eleuterio F. Toro (2009)
  # Riemann Solvers and Numerical Methods for Fluid Dynamics: A Pratical Introduction
  # [DOI: 10.1007/b79761](https://doi.org/10.1007/b79761)
  if v_normal <= 0.0
    sound_speed = sqrt(equations.gamma * p_local / rho_local) # local sound speed
    p_star = p_local * (1.0 + 0.5 * (equations.gamma - 1) * v_normal / sound_speed)^(2.0 * equations.gamma * equations.inv_gamma_minus_one)
  else # v_normal > 0.0
    A = 2.0 / ((equations.gamma + 1) * rho_local)
    B = p_local * (equations.gamma - 1) / (equations.gamma + 1)
    p_star = p_local + 0.5 * v_normal / A * (v_normal + sqrt(v_normal^2 + 4.0 * A * (p_local + B)))
  end

  # For the slip wall we directly set the flux as the normal velocity is zero
  return SVector(zero(eltype(u_inner)),
                 p_star * normal[1],
                 p_star * normal[2],
                 zero(eltype(u_inner))) * norm_
end

"""
    boundary_condition_slip_wall(u_inner, orientation, direction, x, t,
                                 surface_flux_function, equations::CompressibleDryEulerEquations2D)
Should be used together with [`TreeMesh`](@ref).

!!! warning "Experimental code"
    This wall function can change any time.
"""
function boundary_condition_slip_wall(u_inner, orientation, direction, x, t,
                                      surface_flux_function, equations::CompressibleDryEulerEquations2D)
  # get the appropriate normal vector from the orientation
  if orientation == 1
    normal = SVector(1, 0)
  else # orientation == 2
    normal = SVector(0, 1)
  end

  # compute and return the flux using `boundary_condition_slip_wall` routine above
  return boundary_condition_slip_wall(u_inner, normal, x, t, surface_flux_function, equations)
end

"""
    boundary_condition_slip_wall(u_inner, normal_direction, direction, x, t,
                                 surface_flux_function, equations::CompressibleDryEulerEquations2D)
Should be used together with [`StructuredMesh`](@ref).

!!! warning "Experimental code"
    This wall function can change any time.
"""
function boundary_condition_slip_wall(u_inner, normal_direction::AbstractVector, direction, x, t,
                                      surface_flux_function, equations::CompressibleDryEulerEquations2D)
  # flip sign of normal to make it outward pointing, then flip the sign of the normal flux back
  # to be inward pointing on the -x and -y sides due to the orientation convention used by StructuredMesh
  if isodd(direction)
    boundary_flux = -boundary_condition_slip_wall(u_inner, -normal_direction,
                                                  x, t, surface_flux_function, equations)
  else
    boundary_flux = boundary_condition_slip_wall(u_inner, normal_direction,
                                                 x, t, surface_flux_function, equations)
  end

  return boundary_flux
end

# Calculate 1D flux for a single point
@inline function flux(u, orientation::Integer, equations::CompressibleDryEulerEquations2D)
  rho, rho_v1, rho_v2, rho_e = u
  v1 = rho_v1 / rho
  v2 = rho_v2 / rho
  p = (equations.gamma - 1) * (rho_e - 0.5 * (rho_v1 * v1 + rho_v2 * v2))
  if orientation == 1
    f1 = rho_v1
    f2 = rho_v1 * v1 + p
    f3 = rho_v1 * v2
    f4 = (rho_e + p) * v1
  else
    f1 = rho_v2
    f2 = rho_v2 * v1
    f3 = rho_v2 * v2 + p
    f4 = (rho_e + p) * v2
  end
  return SVector(f1, f2, f3, f4)
end


# Calculate 1D flux for a single point in the normal direction
# Note, this directional vector is not normalized
@inline function flux(u, normal_direction::AbstractVector, equations::CompressibleDryEulerEquations2D)
  rho_e = last(u)
  rho, v1, v2, p = cons2prim(u, equations)

  v_normal = v1 * normal_direction[1] + v2 * normal_direction[2]
  rho_v_normal = rho * v_normal
  f1 = rho_v_normal
  f2 = rho_v_normal * v1 + p * normal_direction[1]
  f3 = rho_v_normal * v2 + p * normal_direction[2]
  f4 = (rho_e + p) * v_normal
  return SVector(f1, f2, f3, f4)
end


function initial_condition_gaussian_bubble(x, t, equations::CompressibleDryEulerEquations2D)
  # Gaussian bubble at the center (x0, z0) with a potential Temperature 
  # perturbation of 0.5 K (for a 1x1.5 km^2 box)
  
  #Initial potential temperature
  theta_ini = 303.15
  v1 = 20
  v2 = 0

  # Bubble center (x0, z0) in meters
  x0 = 750
  z0 = 260

  # Distance from the bubble center
  r = sqrt((x[1]-x0)^2 + (x[2]-z0)^2)

  # Scaling parameters
  A = 0.5
  a = 50
  s = 100

  # Potential temperature perturbation 
  if r > a
    theta_pert = A * exp(- inv(s^2) * (r-a)^2)
  else
    theta_pert = A
  end

  # potential temperature
  theta = theta_ini + theta_pert

  pi_exner = 1 - equations.g / (equations.c_p * theta) * x[2]
  rho = equations.p_0 / (equations.R_d * theta) * (pi_exner)^(equations.c_v / equations.R_d)
  p = equations.p_0 * (1 - equations.kappa * equations.g * x[2] / (equations.R_d * theta_ini))^(equations.c_p / equations.R_d)
  T = p / (equations.R_d * rho)

  rho_v1 = rho*v1
  rho_v2 = rho*v2
  rho_e = rho * equations.c_v * T + 0.5 * rho *(v1^2 + v2^2)
  return SVector(rho, rho_v1, rho_v2, rho_e)
end


function source_terms_warm_bubble(du, u, equations::CompressibleDryEulerEquations2D, dg)
  for j in eachnode(dg), i in eachnode(dg)
    # TODO: performance use temp
    #x1 = x[1, i, j, element_id]
    #x2 = x[2, i, j, element_id]
    du[3, i, j, :] -=  equations.g * u[1, i, j, :]
    du[4, i, j, :] -=  equations.g * u[3, i, j, :]
  end
  return nothing
end


@inline function flux_LMARS(u_ll, u_rr, orientation::Integer , equations::CompressibleDryEulerEquations2D)
  @unpack a, gamma = equations
  
  # Unpack left and right state
  rho_ll, rho_v1_ll, rho_v2_ll, rho_e_ll = u_ll
  v1_ll = rho_v1_ll/rho_ll
  v2_ll = rho_v2_ll/rho_v2_ll

  rho_rr, rho_v1_rr, rho_v2_rr, rho_e_rr = u_rr
  v1_rr = rho_v1_rr/rho_rr
  v2_rr = rho_v2_rr/rho_v2_rr
  
  # Compute the necessary interface flux components

  rho_mean = 0.5*(rho_ll + rho_rr) # TODO why choose the mean value here?

  p_ll = (gamma - 1) * (rho_e_ll - 0.5 * (rho_v1_ll * v1_ll + rho_v2_ll * v2_ll))
  p_rr = (gamma - 1) * (rho_e_rr - 0.5 * (rho_v1_rr * v1_rr + rho_v2_rr * v2_rr))

  if orientation == 1

    beta = 0.5 # diffusion parameter <= 1 

    v_interface = 0.5*(v1_rr + v1_ll) - beta*inv(2*rho_mean*a)*(p_rr-p_ll)
    p_interface = 0.5*(p_rr + p_ll) - beta*0.5*rho_mean*a*(v1_rr-v1_ll)

    if (v_interface > 0)
      f1 = rho_ll
      f2 = rho_v1_ll
      f3 = rho_v2_ll
      f4 = rho_e_ll + p_ll
    else
      f1 = rho_rr
      f2 = rho_v1_rr
      f3 = rho_v2_rr
      f4 = rho_e_rr + p_rr
    end

    flux = SVector(f1, f2, f3, f4) *v_interface + SVector(0, 1, 0, 0) * p_interface

  else # orientation = 2

    v_interface = 0.5*(v2_rr + v2_ll) - inv(2*rho_mean*a)*(p_rr-p_ll)
    p_interface = 0.5*(p_rr + p_ll) - 0.5*rho_mean*a*(v2_rr-v2_ll)

    if (v_interface > 0)
      f1 = rho_ll
      f2 = rho_v1_ll
      f3 = rho_v2_ll
      f4 = rho_e_ll + p_ll
    else
      f1 = rho_rr
      f2 = rho_v1_rr
      f3 = rho_v2_rr
      f4 = rho_e_rr + p_rr
    end

    flux = SVector(f1, f2, f3, f4) * v_interface + SVector(0, 0, 1, 0) * p_interface
  end

  return flux
end


@inline function flux_LMARS(u_ll, u_rr, normal_direction::AbstractVector , equations::CompressibleDryEulerEquations2D)
  # Unpack left and right state
  rho_ll, rho_v1_ll, rho_v2_ll, rho_e_ll = u_ll
  rho_rr, rho_v1_rr, rho_v2_rr, rho_e_rr = u_rr

  v1_ll = rho_v1_ll/rho_ll
  v2_ll = rho_v2_ll/rho_v2_ll
  
  v1_rr = rho_v1_rr/rho_rr
  v2_rr = rho_v2_rr/rho_v2_rr
  # Calculate scalar product with normal vector
  v_dot_n_ll = v1_ll * normal_direction[1] + v2_ll * normal_direction[2]
  v_dot_n_rr = v1_rr * normal_direction[1] + v2_rr * normal_direction[2]

  
  # Compute the necessary interface flux components

  rho_mean = 0.5*(rho_ll + rho_rr) # TODO why choose the mean value here?

  v_add = rho_v1 * v1 + rho_v2 * v2
  p_ll = (equations.gamma - 1) * (rho_e_ll - 0.5 * v_add)
  p_rr = (equations.gamma - 1) * (rho_e_rr - 0.5 * v_add)

  # diffusion parameter <= 1 
  beta = 1 

  v_interface = 0.5*(v_dot_n_rr + v_dot_n_ll) - beta*inv(2*rho_mean*equations.a)*(p_rr-p_ll)
  p_interface = 0.5*(p_rr + p_ll) - beta*0.5*rho_mean*equations.a*(v_dot_n_rr-v_dot_n_ll)

  if (v_interface > 0)
    f1 = rho_ll
    f2 = f1*v_dot_n_ll[1]
    f3 = f1*v_dot_n_ll[2]
    f4 = rho_e_ll
  else
    f1 = rho_rr
    f2 = f1*v_dot_n_rr[1]
    f3 = f1*v_dot_n_rr[2]
    f4 = rho_e_rr
  end

  flux = SVector(f1, f2, f3, f4)*v_interface + SVector(0, normal_direction[1], normal_direction[2], 0)*p_interface

  return flux
end



# Called inside `FluxRotated` in `numerical_fluxes.jl` so the direction
# has been normalized prior to this rotation of the state vector
@inline function rotate_to_x(u, normal_vector, equations::CompressibleDryEulerEquations2D)
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
@inline function rotate_from_x(u, normal_vector, equations::CompressibleDryEulerEquations2D)
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


@inline function max_abs_speeds(u, equations::CompressibleDryEulerEquations2D)
  rho, v1, v2, p = cons2prim(u, equations)
  c = sqrt(equations.gamma * p / rho)

  return abs(v1) + c, abs(v2) + c
end


# Convert conservative variables to primitive
@inline function cons2prim(u, equations::CompressibleDryEulerEquations2D)
  rho, rho_v1, rho_v2, rho_e = u

  v1 = rho_v1 / rho
  v2 = rho_v2 / rho
  p = (equations.gamma - 1) * (rho_e - 0.5 * (rho_v1 * v1 + rho_v2 * v2))

  return SVector(rho, v1, v2, p)
end


# Convert conservative variables to entropy
@inline function cons2entropy(u, equations::CompressibleDryEulerEquations2D)
  rho, rho_v1, rho_v2, rho_e = u

  v1 = rho_v1 / rho
  v2 = rho_v2 / rho
  v_square = v1^2 + v2^2
  p = (equations.gamma - 1) * (rho_e - 0.5 * rho * v_square)

  s = log(p) - equations.gamma*log(rho)
  rho_p = rho / p

  w1 = (equations.gamma - s) * inv(equations.gamma - 1) - 0.5 * rho_p * v_square
  w2 = rho_p * v1
  w3 = rho_p * v2
  w4 = -rho_p

  return SVector(w1, w2, w3, w4)
end

@inline function entropy2cons(w, equations::CompressibleDryEulerEquations2D)
  # See Hughes, Franca, Mallet (1986) A new finite element formulation for CFD
  # [DOI: 10.1016/0045-7825(86)90127-1](https://doi.org/10.1016/0045-7825(86)90127-1)
  @unpack gamma = equations

  # convert to entropy `-rho * s` used by Hughes, France, Mallet (1986)
  # instead of `-rho * s / (gamma - 1)`
  V1, V2, V3, V5 = w .* (gamma-1)

  # s = specific entropy, eq. (53)
  s = gamma - V1 + (V2^2 + V3^2)/(2*V5)

  # eq. (52)
  rho_iota = ((gamma-1) / (-V5)^gamma)^(equations.inv_gamma_minus_one)*exp(-s * equations.inv_gamma_minus_one)

  # eq. (51)
  rho      = -rho_iota * V5
  rho_v1   =  rho_iota * V2
  rho_v2   =  rho_iota * V3
  rho_e    =  rho_iota * (1-(V2^2 + V3^2)/(2*V5))
  return SVector(rho, rho_v1, rho_v2, rho_e)
end


# Convert primitive to conservative variables
@inline function prim2cons(prim, equations::CompressibleDryEulerEquations2D)
  rho, v1, v2, p = prim
  rho_v1 = rho * v1
  rho_v2 = rho * v2
  rho_e  = p * equations.inv_gamma_minus_one + 0.5 * (rho_v1 * v1 + rho_v2 * v2)
  return SVector(rho, rho_v1, rho_v2, rho_e)
end

@inline function cons2pot(u, equation::CompressibleDryEulerEquations2D)
  rho, rho_v1, rho_v2, rho_e = u

  v1 = rho_v1 / rho
  v2 = rho_v2 / rho

  pot1 = rho
  pot2 = v1
  pot3 = v2
  pot4 = equation.p_0 * (((equation.gamma - 1) * (rho_e - 1/2 * (rho_v1 * v1 + rho_v2 * v2)))
                        / equation.p_0)^(1-equation.kappa) / (equation.R_d * rho)

  return SVector(pot1, pot2, pot3, pot4)
end


@inline function density(u, equations::CompressibleDryEulerEquations2D)
 rho = u[1]
 return rho
end


@inline function pressure(u, equations::CompressibleDryEulerEquations2D)
 rho, rho_v1, rho_v2, rho_e = u
 p = (equations.gamma - 1) * (rho_e - 0.5 * (rho_v1^2 + rho_v2^2) / rho)
 return p
end


@inline function density_pressure(u, equations::CompressibleDryEulerEquations2D)
 rho, rho_v1, rho_v2, rho_e = u
 rho_times_p = (equations.gamma - 1) * (rho * rho_e - 0.5 * (rho_v1^2 + rho_v2^2))
 return rho_times_p
end


# Calculate thermodynamic entropy for a conservative state `cons`
@inline function entropy_thermodynamic(cons, equations::CompressibleDryEulerEquations2D)
  # Pressure
  p = (equations.gamma - 1) * (cons[4] - 1/2 * (cons[2]^2 + cons[3]^2) / cons[1])

  # Thermodynamic entropy
  s = log(p) - equations.gamma*log(cons[1])

  return s
end


# Calculate mathematical entropy for a conservative state `cons`
@inline function entropy_math(cons, equations::CompressibleDryEulerEquations2D)
  # Mathematical entropy
  S = -entropy_thermodynamic(cons, equations) * cons[1] * equations.inv_gamma_minus_one

  return S
end


# Default entropy is the mathematical entropy
@inline entropy(cons, equations::CompressibleDryEulerEquations2D) = entropy_math(cons, equations)


# Calculate total energy for a conservative state `cons`
@inline energy_total(cons, ::CompressibleDryEulerEquations2D) = cons[4]


# Calculate kinetic energy for a conservative state `cons`
@inline function energy_kinetic(u, equations::CompressibleDryEulerEquations2D)
  rho, rho_v1, rho_v2, rho_e = u
  return (rho_v1^2 + rho_v2^2) / (2 * rho)
end


# Calculate internal energy for a conservative state `cons`
@inline function energy_internal(cons, equations::CompressibleDryEulerEquations2D)
  return energy_total(cons, equations) - energy_kinetic(cons, equations)
end


end # @muladd
