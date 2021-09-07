@muladd begin

struct CompressiblePotTempEulerEquations2D{RealT<:Real} <: AbstractCompressiblePotTempEulerEquations{2, 4}
  p_0::RealT   # constant reference pressure 1000 hPa(100000 Pa)
  c_p::RealT
  c_v::RealT
  R_d::RealT   # gas constant
  g::RealT # gravitation constant
  kappa::RealT # ratio of the gas constand R_d
  gamma::RealT # = inv(kappa- 1); can be used to write slow divisions as fast multiplications
end

function CompressiblePotTempEulerEquations2D(;RealT=Float64)
   p_0 = 100000.0
   c_p = 1004.0
   c_v = 717.0
   R_d = c_p-c_v 
   g = 9.81
   gamma = c_p / c_v
   kappa = 1 - inv(gamma)

   return CompressiblePotTempEulerEquations2D{RealT}(p_0, c_p, c_v, R_d, g, kappa, gamma)
  end

varnames(::typeof(cons2cons), ::CompressiblePotTempEulerEquations2D) = ("rho", "rho_v1", "rho_v2", "rho_theta")


function initial_condition_gaussian_bubble(x, t, equations::CompressiblePotTempEulerEquations2D)
  # Gaussian bubble at the center (x0, z0) with a potential Temperature 
  # perturbation of 0.5 K (for a 1x1.5 km^2 box)
  
  #Initial potential temperature
  theta_ini = 303.15
  v1 = 0
  v2 = 0

  # Bubble center (x0, y0) in meters
  x0 = 500
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

  pi_exner = 1-equations.g / (equations.c_p * theta) * x[2]
  rho = equations.p_0 / (equations.R_d * theta) * (pi_exner)^(equations.c_v / equations.R_d)
  rho_v1 = rho*v1
  rho_v2 = rho*v2
  theta_hat = rho * theta / (equations.p_0)^equations.kappa
  
  return SVector(rho, rho_v1, rho_v2, theta_hat)
end


function boundary_condition_reflection(u_inner, orientation, direction, x, t,
  surface_flux_function,
  equations::CompressiblePotTempEulerEquations2D)
  # Orientation 3 neg-y-direction/unten
  # Orientation 4 pos-y-direction/oben
  if orientation != (3 || 4)
    error("This boundary condition is not supposed to be called in x direction")
  end
  rho, rho_v1, rho_v2, theta_hat = u_inner

  p_prime = 0
  
  p = (equations.R_d * theta_hat)^equations.gamma
  a = sqrt(equations.gamma*p*inv(rho))

return Svector(0, 0, p_prime - a*inv(rho)*rho_v1, 0)
end


# Calculate 1D flux for a single point
@inline function flux(u, orientation::Integer, equations::CompressiblePotTempEulerEquations2D)
  rho, rho_v1, rho_v2, theta_hat = u
  v1 = rho_v1/rho
  v2 = rho_v2/rho
  p = (equations.R_d * theta_hat)^equations.gamma
  p_prime=0 # This assumes a hydrostatic system TODO: add p_prime=p=p_star

  if orientation == 1
    f1 = rho_v1
    f2 = rho_v1 * v1 + p
    f3 = rho_v1 * v2
    f4 = (theta_hat) * v1
  else
    f1 = rho_v2
    f2 = rho_v2 * v1
    f3 = rho_v2 * v2 + p_prime
    f4 = (theta_hat) * v2
  end
  return SVector(f1, f2, f3, f4)
end


@inline function flux_LMARS(u_ll, u_rr, orientation::Integer , equations::CompressiblePotTempEulerEquations2D)
  # Unpack left and right state
  rho_ll, rho_v1_ll, rho_v2_ll, theta_hat_ll = u_ll
  v1_ll = rho_v1_ll/rho_ll
  v2_ll = rho_v2_ll/rho_v2_ll

  p_ll = (equations.R_d * theta_hat_ll)^equations.gamma
  p_prime_ll = 0

  rho_rr, rho_v1_rr, rho_v2_rr, theta_hat_rr = u_rr
  v1_rr = rho_v1_rr/rho_rr
  v2_rr = rho_v2_rr/rho_v2_rr
  
  p_rr = (equations.R_d * theta_hat_rr)^equations.gamma
  p_prime_rr = 0
  
  # Compute the necessary interface flux components

  rho_mean = 0.5*(rho_ll + rho_rr) # TODO why choose the mean value here?
  p_mean = 0.5*(p_ll + p_rr) # TODO does one choose the mean value here? 

  a = sqrt(equations.gamma*p_mean*inv(rho_mean))

  if orientation == 1

    beta = 1 # diffusion parameter <= 1 

    v_interface = 0.5*(v1_rr + v1_ll) - beta*inv(2*rho_mean*a)*(p_rr-p_ll)
    p_interface = 0.5*(p_rr + p_ll) - beta*0.5*rho_mean*a*(v1_rr-v1_ll)

    if (v_interface > 0)
      f1 = rho_ll
      f2 = rho_v1_ll
      f3 = rho_v2_ll
      f4 = theta_hat_ll
    else
      f1 = rho_rr
      f2 = rho_v1_rr
      f3 = rho_v2_rr
      f4 = theta_hat_rr
    end

    flux= SVector(f1, f2, f3, f4)*v_interface + SVector(0, 1, 0, 0)*p_interface
  
  else

    v_interface = 0.5*(v2_rr + v2_ll) - inv(2*rho_mean*a)*(p_prime_rr-p_prime_ll)
    p_prime_interface = 0.5*(p_prime_rr + p_prime_ll) - 0.5*rho_mean*a*(v2_rr-v2_ll)

    if (v_interface > 0)
      f1 = rho_ll
      f2 = rho_v1_ll
      f3 = rho_v2_ll
      f4 = theta_hat_ll
    else
      f1 = rho_rr
      f2 = rho_v1_rr
      f3 = rho_v2_rr
      f4 = theta_hat_rr
    end

    flux= SVector(f1, f2, f3, f4)*v_interface + SVector(0, 1, 0, 0)*p_prime_interface
  end

  return flux
end



# Called inside `FluxRotated` in `numerical_fluxes.jl` so the direction
# has been normalized prior to this rotation of the state vector
@inline function rotate_to_x(u, normal_vector, equations::CompressiblePotTempEulerEquations2D)
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
@inline function rotate_from_x(u, normal_vector, equations::CompressiblePotTempEulerEquations2D)
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

@inline function max_abs_speeds(u, equations::CompressiblePotTempEulerEquations2D)
  rho, rho_v1, rho_v2, theta_hat = u
  v_1 = rho_v1 * inv(rho)
  v_2 = rho_v2 * inv(rho)
  p = (equations.R_d * theta_hat)^equations.gamma
  a = sqrt(equations.gamma * p / rho)

  return abs(v1) + a, abs(v2) + a
end


end # @muladd
