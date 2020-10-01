@doc raw"""
    CompressibleEulerEquations1D

The compressible Euler equations for an ideal gas in one space dimension.
"""
struct CompressibleEulerEquations1D <: AbstractCompressibleEulerEquations{1, 3}
  gamma::Float64
end

function CompressibleEulerEquations1D()
  gamma = parameter("gamma", 1.4)

  CompressibleEulerEquations1D(gamma)
end


get_name(::CompressibleEulerEquations1D) = "CompressibleEulerEquations1D"
varnames_cons(::CompressibleEulerEquations1D) = @SVector ["rho", "rho_v1", "rho_e"]
varnames_prim(::CompressibleEulerEquations1D) = @SVector ["rho", "v1", "p"]


# Set initial conditions at physical location `x` for time `t`
function initial_conditions_density_pulse(x, t, equation::CompressibleEulerEquations1D)
  rho = 1 + exp(-(x[1]^2 ))/2
  v1 = 1
  rho_v1 = rho * v1
  p = 1
  rho_e = p/(equation.gamma - 1) + 1/2 * rho * v1^2
  return @SVector [rho, rho_v1, rho_e]
end

"""
    initial_conditions_density_wave(x, t, equation::CompressibleEulerEquations1D)

test case for stability of EC fluxes from paper: https://arxiv.org/pdf/2007.09026.pdf
domain [-1, 1]
mesh = 4x4, polydeg = 5
"""

function initial_conditions_density_wave(x, t, equation::CompressibleEulerEquations1D)
  v1 = 0.1
  rho = 1 + 0.98 * sinpi(2 * (x[1] - t * v1))
  rho_v1 = rho * v1
  p = 20
  rho_e = p / (equation.gamma - 1) + 1/2 * rho * v1^2
  return @SVector [rho, rho_v1, rho_e]
end

function initial_conditions_constant(x, t, equation::CompressibleEulerEquations1D)
  rho = 1.0
  rho_v1 = 0.1
  rho_e = 10.0
  return @SVector [rho, rho_v1, rho_e]
end

function initial_conditions_convergence_test(x, t, equation::CompressibleEulerEquations1D)
  c = 2
  A = 0.1
  L = 2
  f = 1/L
  ω = 2 * pi * f
  ini = c + A * sin(ω * (x[1] - t))

  rho = ini
  rho_v1 = ini
  rho_e = ini^2

  return @SVector [rho, rho_v1, rho_e]
end


function initial_conditions_weak_blast_wave(x, t, equation::CompressibleEulerEquations1D)
  # From Hennemann & Gassner JCP paper 2020 (Sec. 6.3)
  # Set up polar coordinates
  inicenter = SVector(0.0, 0.0)
  x_norm = x[1] - inicenter[1]
  r = abs(x_norm)
  phi = atan(0.0, x_norm)
  cos_phi = cos(phi)

  # Calculate primitive variables
  rho = r > 0.5 ? 1.0 : 1.1691
  v1  = r > 0.5 ? 0.0 : 0.1882 * cos_phi
  p   = r > 0.5 ? 1.0 : 1.245

  return prim2cons(SVector(rho, v1, p), equation)
end

function initial_conditions_blast_wave(x, t, equation::CompressibleEulerEquations1D)
  # Modified From Hennemann & Gassner JCP paper 2020 (Sec. 6.3) -> "medium blast wave"
  # Set up polar coordinates
  inicenter = SVector(0.0, 0.0)
  x_norm = x[1] - inicenter[1]
  r = abs(x_norm)
  phi = atan(0.0, x_norm)
  cos_phi = cos(phi)

  # Calculate primitive variables
  rho = r > 0.5 ? 1.0 : 1.1691
  v1  = r > 0.5 ? 0.0 : 0.1882 * cos_phi
  p   = r > 0.5 ? 1.0E-3 : 1.245

  return prim2cons(SVector(rho, v1, p), equation)
end

function initial_conditions_sedov_blast_wave(x, t, equation::CompressibleEulerEquations1D)
  # Set up polar coordinates
  inicenter = SVector(0.0, 0.0)
  x_norm = x[1] - inicenter[1]
  r = abs(x_norm)

  # Setup based on http://flash.uchicago.edu/site/flashcode/user_support/flash_ug_devel/node184.html#SECTION010114000000000000000
  r0 = 0.21875 # = 3.5 * smallest dx (for domain length=4 and max-ref=6)
  # r0 = 0.5 # = more reasonable setup
  E = 1.0
  p0_inner = 6 * (equation.gamma - 1) * E / (3 * pi * r0)
  p0_outer = 1.0e-5 # = true Sedov setup
  # p0_outer = 1.0e-3 # = more reasonable setup

  # Calculate primitive variables
  rho = 1.0
  v1  = 0.0
  p   = r > r0 ? p0_outer : p0_inner

  return prim2cons(SVector(rho, v1, p), equation)
end

# Apply boundary conditions
function boundary_conditions_convergence_test(u_inner, orientation, direction, x, t,
                                              surface_flux_function,
                                              equation::CompressibleEulerEquations1D)
  u_boundary = initial_conditions_convergence_test(x, t, equation)

  # Calculate boundary flux
  if direction == 2 # u_inner is "left" of boundary, u_boundary is "right" of boundary
    flux = surface_flux_function(u_inner, u_boundary, orientation, equation)
  else # u_boundary is "left" of boundary, u_inner is "right" of boundary
    flux = surface_flux_function(u_boundary, u_inner, orientation, equation)
  end

  return flux
end

function boundary_conditions_sedov_self_gravity(u_inner, orientation, direction, x, t,
                                                surface_flux_function,
                                                equation::CompressibleEulerEquations1D)
  # velocities are zero, density/pressure are ambient values according to
  # initial_conditions_sedov_self_gravity
  rho = 1e-5
  v1 = 0.0
  p = 1e-5

  u_boundary = prim2cons(SVector(rho, v1, p), equation)

  # Calculate boundary flux
  if direction == 2  # u_inner is "left" of boundary, u_boundary is "right" of boundary
    flux = surface_flux_function(u_inner, u_boundary, orientation, equation)
  else # u_boundary is "left" of boundary, u_inner is "right" of boundary
    flux = surface_flux_function(u_boundary, u_inner, orientation, equation)
  end

  return flux
end

# Apply source terms
function source_terms_convergence_test(ut, u, x, element_id, t, n_nodes, equation::CompressibleEulerEquations1D)
  # Same settings as in `initial_conditions`
  c = 2
  A = 0.1
  L = 2
  f = 1/L
  ω = 2 * pi * f
  γ = equation.gamma

  for i in 1:n_nodes
    x1 = x[1, i, element_id]

    si, co = sincos((t - x1)*ω)
    tmp = (-((4 * si * A - 4c) + 1) * (γ - 1) * co * A * ω) / 2

    ut[2, i, element_id] += tmp
    ut[3, i, element_id] += tmp

    # Original terms (without performanc enhancements)
    # ut[1, i, element_id] += 0
    # ut[2, i, element_id] += (-(((4 * sin((t - x1) * ω) * A - 4c) + 1)) *
    #                          (γ - 1) * cos((t - x1) * ω) * A * ω) / 2
    # ut[3, i, element_id] += (-(((4 * sin((t - x1) * ω) * A - 4c) + 1)) *
    #                          (γ - 1) * cos((t - x1) * ω) * A * ω) / 2
  end

  return nothing
end
#=
function source_terms_eoc_test_coupled_euler_gravity(ut, u, x, element_id, t, n_nodes, equation::CompressibleEulerEquations2D)
  # Same settings as in `initial_conditions_eoc_test_coupled_euler_gravity`
  c = 2.0
  A = 0.1
  G = 1.0 # gravitational constant, must match coupling solver
  C_grav = -2.0 * G / pi # 2 == 4 / ndims

  for j in 1:n_nodes, i in 1:n_nodes
    x1 = x[1, i, j, element_id]
    x2 = x[2, i, j, element_id]
    si, co = sincos(pi * (x1 + x2 - t))
    rhox = A * pi * co
    rho  = c + A *  si

    ut[1, i, j, element_id] += rhox
    ut[2, i, j, element_id] += rhox
    ut[3, i, j, element_id] += rhox
    ut[4, i, j, element_id] += (1.0 - C_grav*rho)*rhox
  end

  return nothing
end

function source_terms_eoc_test_euler(ut, u, x, element_id, t, n_nodes, equation::CompressibleEulerEquations2D)
  # Same settings as in `initial_conditions_eoc_test_coupled_euler_gravity`
  c = 2.0
  A = 0.1
  G = 1.0
  C_grav = -2 * G / pi # 2 == 4 / ndims

  for j in 1:n_nodes, i in 1:n_nodes
    x1 = x[1, i, j, element_id]
    x2 = x[2, i, j, element_id]
    si, co = sincos(pi * (x1 + x2 - t))
    rhox = A * pi * co
    rho  = c + A *  si

    ut[1, i, j, element_id] += rhox
    ut[2, i, j, element_id] += rhox * (1 -     C_grav * rho)
    ut[3, i, j, element_id] += rhox * (1 -     C_grav * rho)
    ut[4, i, j, element_id] += rhox * (1 - 3 * C_grav * rho)
  end

  return nothing
end
=#
# Empty source terms required for coupled Euler-gravity simulations
function source_terms_harmonic(ut, u, x, element_id, t, n_nodes, equation::CompressibleEulerEquations1D)
  # OBS! used for the Jeans instability as well as self-gravitating Sedov blast
  # TODO: make this cleaner and let each solver have a different source term name
  return nothing
end

# Calculate 1D flux for a single point
@inline function calcflux(u, orientation, equation::CompressibleEulerEquations1D)
  rho, rho_v1, rho_e = u
  v1 = rho_v1/rho
  p = (equation.gamma - 1) * (rho_e - 1/2 * rho * v1^2)
  # Ignore orientation since it is always "1" in 1D
  f1 = rho_v1
  f2 = rho_v1 * v1 + p
  f3 = (rho_e + p) * v1
  return SVector(f1, f2, f3)
end


"""
    function flux_shima_etal(u_ll, u_rr, orientation, equation::CompressibleEulerEquations1D)

This flux is is a modification of the original kinetic energy preserving two-point flux by
Kuya, Totani and Kawai (2018)
  Kinetic energy and entropy preserving schemes for compressible flows
  by split convective forms
  [DOI: 10.1016/j.jcp.2018.08.058](https://doi.org/10.1016/j.jcp.2018.08.058)
The modification is in the energy flux to guarantee pressure equilibrium and was developed by
Nao Shima, Yuichi Kuya, Yoshiharu Tamaki, Soshi Kawai (JCP 2020)
  Preventing spurious pressure oscillations in split convective form discretizations for
  compressible flows
"""
@inline function flux_shima_etal(u_ll, u_rr, orientation, equation::CompressibleEulerEquations1D)
  # Unpack left and right state
  rho_ll, rho_v1_ll, rho_e_ll = u_ll
  rho_rr, rho_v1_rr, rho_e_rr = u_rr

  v1_ll = rho_v1_ll / rho_ll
  v1_rr = rho_v1_rr / rho_rr
  p_ll  = (equation.gamma - 1) * (rho_e_ll - 1/2 * rho_ll * (v1_ll^2))
  p_rr  = (equation.gamma - 1) * (rho_e_rr - 1/2 * rho_rr * (v1_rr^2))

  # Average each factor of products in flux
  rho_avg = 1/2 * (rho_ll + rho_rr)
  v1_avg  = 1/2 * ( v1_ll +  v1_rr)
  p_avg   = 1/2 * (  p_ll +   p_rr)
  kin_avg = 1/2 * (v1_ll*v1_rr)

  # Calculate fluxes
  # Ignore orientation since it is always "1" in 1D
  pv1_avg = 1/2 * (p_ll*v1_rr + p_rr*v1_ll)
  f1 = rho_avg * v1_avg
  f2 = rho_avg * v1_avg * v1_avg + p_avg
  f3 = p_avg*v1_avg/(equation.gamma-1) + rho_avg*v1_avg*kin_avg + pv1_avg

  return SVector(f1, f2, f3)
end


"""
    flux_kennedy_gruber(u_ll, u_rr, orientation, equation::CompressibleEulerEquations1D)

Kinetic energy preserving two-point flux by Kennedy and Gruber (2008)
  Reduced aliasing formulations of the convective terms within the
  Navier-Stokes equations for a compressible fluid
[DOI: 10.1016/j.jcp.2007.09.020](https://doi.org/10.1016/j.jcp.2007.09.020)
"""
@inline function flux_kennedy_gruber(u_ll, u_rr, orientation, equation::CompressibleEulerEquations1D)
  # Unpack left and right state
  rho_ll, rho_v1_ll, rho_e_ll = u_ll
  rho_rr, rho_v1_rr, rho_e_rr = u_rr

  v1_ll = rho_v1_ll/rho_ll
  v1_rr = rho_v1_rr/rho_rr

  # Average each factor of products in flux
  rho_avg = 1/2 * (rho_ll + rho_rr)
  v1_avg = 1/2 * (v1_ll + v1_rr)
  p_avg = 1/2 * ((equation.gamma - 1) * (rho_e_ll - 1/2 * rho_ll * (v1_ll^2)) +
                 (equation.gamma - 1) * (rho_e_rr - 1/2 * rho_rr * (v1_rr^2)))
  e_avg = 1/2 * (rho_e_ll/rho_ll + rho_e_rr/rho_rr)

  # Ignore orientation since it is always "1" in 1D
  f1 = rho_avg * v1_avg
  f2 = rho_avg * v1_avg * v1_avg + p_avg
  f3 = (rho_avg * e_avg + p_avg) * v1_avg

  return SVector(f1, f2, f3)
end


"""
    flux_chandrashekar(u_ll, u_rr, orientation, equation::CompressibleEulerEquations2D)

Entropy conserving two-point flux by Chandrashekar (2013)
  Kinetic Energy Preserving and Entropy Stable Finite Volume Schemes
  for Compressible Euler and Navier-Stokes Equations
[DOI: 10.4208/cicp.170712.010313a](https://doi.org/10.4208/cicp.170712.010313a)
"""
@inline function flux_chandrashekar(u_ll, u_rr, orientation, equation::CompressibleEulerEquations1D)
  # Unpack left and right state
  rho_ll, rho_v1_ll, rho_e_ll = u_ll
  rho_rr, rho_v1_rr, rho_e_rr = u_rr

  v1_ll = rho_v1_ll/rho_ll
  v1_rr = rho_v1_rr/rho_rr
  p_ll =  (equation.gamma - 1) * (rho_e_ll - 1/2 * rho_ll * (v1_ll^2))
  p_rr =  (equation.gamma - 1) * (rho_e_rr - 1/2 * rho_rr * (v1_rr^2))
  beta_ll = 0.5*rho_ll/p_ll
  beta_rr = 0.5*rho_rr/p_rr
  specific_kin_ll = 0.5*(v1_ll^2)
  specific_kin_rr = 0.5*(v1_rr^2)

  # Compute the necessary mean values
  rho_avg  = 0.5*(rho_ll+rho_rr)
  rho_mean = ln_mean(rho_ll,rho_rr)
  beta_mean = ln_mean(beta_ll,beta_rr)
  beta_avg = 0.5*(beta_ll+beta_rr)
  v1_avg = 0.5*(v1_ll+v1_rr)
  p_mean = 0.5*rho_avg/beta_avg
  velocity_square_avg = specific_kin_ll + specific_kin_rr

  # Calculate fluxes
  # Ignore orientation since it is always "1" in 1D
  f1 = rho_mean * v1_avg
  f2 = f1 * v1_avg + p_mean
  f3 = f1 * 0.5*(1/(equation.gamma-1)/beta_mean - velocity_square_avg)+f2*v1_avg

  return SVector(f1, f2, f3)
end


"""
    flux_ranocha(u_ll, u_rr, orientation, equation::CompressibleEulerEquations2D)

Entropy conserving and kinetic energy preserving two-point flux by Ranocha (2018)
  Generalised Summation-by-Parts Operators and Entropy Stability of Numerical Methods
  for Hyperbolic Balance Laws
[PhD thesis, TU Braunschweig](https://cuvillier.de/en/shop/publications/7743)
See also Ranocha (2020)
  Entropy Conserving and Kinetic Energy Preserving Numerical Methods for
  the Euler Equations Using Summation-by-Parts Operators
[Proceedings of ICOSAHOM 2018](https://doi.org/10.1007/978-3-030-39647-3_42)
"""
@inline function flux_ranocha(u_ll, u_rr, orientation, equation::CompressibleEulerEquations1D)
  # Unpack left and right state
  rho_ll, rho_v1_ll, rho_e_ll = u_ll
  rho_rr, rho_v1_rr, rho_e_rr = u_rr

  v1_ll = rho_v1_ll / rho_ll
  v1_rr = rho_v1_rr / rho_rr
  p_ll =  (equation.gamma - 1) * (rho_e_ll - 0.5 * rho_ll * (v1_ll^2))
  p_rr =  (equation.gamma - 1) * (rho_e_rr - 0.5 * rho_rr * (v1_rr^2 ))

  # Compute the necessary mean values
  rho_mean   = ln_mean(rho_ll, rho_rr)
  rho_p_mean = ln_mean(rho_ll / p_ll, rho_rr / p_rr)
  v1_avg = 0.5 * (v1_ll + v1_rr)
  p_avg  = 0.5 * (p_ll + p_rr)
  velocity_square_avg = 0.5 * (v1_ll*v1_rr)

  # Calculate fluxes
  # Ignore orientation since it is always "1" in 1D
  f1 = rho_mean * v1_avg
  f2 = f1 * v1_avg + p_avg
  f3 = f1 * ( velocity_square_avg + 1 / ((equation.gamma-1) * rho_p_mean) ) + 0.5 * (p_ll*v1_rr + p_rr*v1_ll)

  return SVector(f1, f2, f3)
end


function flux_lax_friedrichs(u_ll, u_rr, orientation, equation::CompressibleEulerEquations1D)
  # Calculate primitive variables and speed of sound
  rho_ll, rho_v1_ll, rho_e_ll = u_ll
  rho_rr, rho_v1_rr, rho_e_rr = u_rr

  v1_ll = rho_v1_ll / rho_ll
  v_mag_ll = abs(v1_ll)
  p_ll = (equation.gamma - 1) * (rho_e_ll - 1/2 * rho_ll * v_mag_ll^2)
  c_ll = sqrt(equation.gamma * p_ll / rho_ll)
  v1_rr = rho_v1_rr / rho_rr
  v_mag_rr = abs(v1_rr)
  p_rr = (equation.gamma - 1) * (rho_e_rr - 1/2 * rho_rr * v_mag_rr^2)
  c_rr = sqrt(equation.gamma * p_rr / rho_rr)

  # Obtain left and right fluxes
  f_ll = calcflux(u_ll, orientation, equation)
  f_rr = calcflux(u_rr, orientation, equation)

  λ_max = max(v_mag_ll, v_mag_rr) + max(c_ll, c_rr)
  f1 = 1/2 * (f_ll[1] + f_rr[1]) - 1/2 * λ_max * (rho_rr    - rho_ll)
  f2 = 1/2 * (f_ll[2] + f_rr[2]) - 1/2 * λ_max * (rho_v1_rr - rho_v1_ll)
  f3 = 1/2 * (f_ll[3] + f_rr[3]) - 1/2 * λ_max * (rho_e_rr  - rho_e_ll)

  return SVector(f1, f2, f3)
end


function flux_hll(u_ll, u_rr, orientation, equation::CompressibleEulerEquations1D)
  # Calculate primitive variables and speed of sound
  rho_ll, rho_v1_ll, rho_e_ll = u_ll
  rho_rr, rho_v1_rr, rho_e_rr = u_rr

  v1_ll = rho_v1_ll / rho_ll
  p_ll = (equation.gamma - 1) * (rho_e_ll - 1/2 * rho_ll * v1_ll^2)

  v1_rr = rho_v1_rr / rho_rr
  p_rr = (equation.gamma - 1) * (rho_e_rr - 1/2 * rho_rr * v1_rr^2)

  # Obtain left and right fluxes
  f_ll = calcflux(u_ll, orientation, equation)
  f_rr = calcflux(u_rr, orientation, equation)

  Ssl = v1_ll - sqrt(equation.gamma * p_ll / rho_ll)
  Ssr = v1_rr + sqrt(equation.gamma * p_rr / rho_rr)

  if Ssl >= 0.0 && Ssr > 0.0
    f1 = f_ll[1]
    f2 = f_ll[2]
    f3 = f_ll[3]
  elseif Ssr <= 0.0 && Ssl < 0.0
    f1 = f_rr[1]
    f2 = f_rr[2]
    f3 = f_rr[3]
  else
    f1 = (Ssr*f_ll[1] - Ssl*f_rr[1] + Ssl*Ssr*(rho_rr[1]    - rho_ll[1]))/(Ssr - Ssl)
    f2 = (Ssr*f_ll[2] - Ssl*f_rr[2] + Ssl*Ssr*(rho_v1_rr[1] - rho_v1_ll[1]))/(Ssr - Ssl)
    f3 = (Ssr*f_ll[3] - Ssl*f_rr[3] + Ssl*Ssr*(rho_e_rr[1]  - rho_e_ll[1]))/(Ssr - Ssl)
  end

  return SVector(f1, f2, f3)
end


# Determine maximum stable time step based on polynomial degree and CFL number
function calc_max_dt(u, element_id, invjacobian, cfl,
                     equation::CompressibleEulerEquations1D, dg)
  λ_max = 0.0
  for i in 1:nnodes(dg)
    rho, rho_v1, rho_e = get_node_vars(u, dg, i, element_id)
    v1 = rho_v1 / rho
    v_mag = abs(v1)
    p = (equation.gamma - 1) * (rho_e - 1/2 * rho * v_mag^2)
    c = sqrt(equation.gamma * p / rho)
    λ_max = max(λ_max, v_mag + c)
  end

  dt = cfl * 2 / (nnodes(dg) * invjacobian * λ_max)

  return dt
end


# Convert conservative variables to primitive
@inline function cons2prim(u, equation::CompressibleEulerEquations1D)
  rho, rho_v1, rho_e = u

  v1 = rho_v1 / rho
  p = (equation.gamma - 1) * (rho_e - 0.5 * rho * (v1^2))

  return SVector(rho, v1, p)
end


# Convert conservative variables to entropy
@inline function cons2entropy(u, equation::CompressibleEulerEquations1D)
  rho, rho_v1, rho_e = u

  v1 = rho_v1 / rho
  v_square = v1^2
  p = (equation.gamma - 1) * (rho_e - 0.5 * rho * v_square)
  s = log(p) - equation.gamma*log(rho)
  rho_p = rho / p

  w1 = (equation.gamma - s) / (equation.gamma-1) - 0.5 * rho_p * v_square
  w2 = rho_p * v1
  w3 = -rho_p

  return SVector(w1, w2, w3)
end


# Convert primitive to conservative variables
@inline function prim2cons(prim, equation::CompressibleEulerEquations1D)
  rho, v1, p = prim
  rho_v1 = rho * v1
  rho_e  = p/(equation.gamma-1) + 0.5 * (rho_v1 * v1)
  return SVector(rho, rho_v1, rho_e)
end


@inline function density(u, equation::CompressibleEulerEquations1D)
 rho = u[1]
 return rho
end

@inline function pressure(u, equation::CompressibleEulerEquations1D)
 rho, rho_v1, rho_e = u
 p = (equation.gamma - 1) * (rho_e - 0.5 * (rho_v1^2) / rho)
 return p
end


@inline function density_pressure(u, equation::CompressibleEulerEquations1D)
 rho, rho_v1, rho_e = u
 rho_times_p = (equation.gamma - 1) * (rho * rho_e - 0.5 * (rho_v1^2))
 return rho_times_p
end


# Calculates the entropy flux in direction "orientation" and the entropy variables for a state cons
# NOTE: This method seems to work currently (b82534e) but is never used anywhere. Thus it is
# commented here until someone uses it or writes a test for it.
# @inline function cons2entropyvars_and_flux(gamma::Float64, cons, orientation::Int)
#   entropy = MVector{4, Float64}(undef)
#   v = (cons[2] / cons[1] , cons[3] / cons[1])
#   v_square= v[1]*v[1]+v[2]*v[2]
#   p = (gamma - 1) * (cons[4] - 1/2 * (cons[2] * v[1] + cons[3] * v[2]))
#   rho_p = cons[1] / p
#   # thermodynamic entropy
#   s = log(p) - gamma*log(cons[1])
#   # mathematical entropy
#   S = - s*cons[1]/(gamma-1)
#   # entropy variables
#   entropy[1] = (gamma - s)/(gamma-1) - 0.5*rho_p*v_square
#   entropy[2] = rho_p*v[1]
#   entropy[3] = rho_p*v[2]
#   entropy[4] = -rho_p
#   # entropy flux
#   entropy_flux = S*v[orientation]
#   return entropy, entropy_flux
# end


# Calculate thermodynamic entropy for a conservative state `cons`
@inline function entropy_thermodynamic(cons, equation::CompressibleEulerEquations1D)
  # Pressure
  p = (equation.gamma - 1) * (cons[3] - 1/2 * (cons[2]^2) / cons[1])

  # Thermodynamic entropy
  s = log(p) - equation.gamma*log(cons[1])

  return s
end


# Calculate mathematical entropy for a conservative state `cons`
@inline function entropy_math(cons, equation::CompressibleEulerEquations1D)
  # Mathematical entropy
  S = -entropy_thermodynamic(cons, equation) * cons[1] / (equation.gamma - 1)

  return S
end


# Default entropy is the mathematical entropy
@inline entropy(cons, equation::CompressibleEulerEquations1D) = entropy_math(cons, equation)


# Calculate total energy for a conservative state `cons`
@inline energy_total(cons, ::CompressibleEulerEquations1D) = cons[3]


# Calculate kinetic energy for a conservative state `cons`
@inline function energy_kinetic(cons, equation::CompressibleEulerEquations1D)
  return 0.5 * (cons[2]^2)/cons[1]
end


# Calculate internal energy for a conservative state `cons`
@inline function energy_internal(cons, equation::CompressibleEulerEquations1D)
  return energy_total(cons, equation) - energy_kinetic(cons, equation)
end
