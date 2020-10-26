
@doc raw"""
    CompressibleEulerEquations3D

The compressible Euler equations for an ideal gas in three space dimensions.
"""
struct CompressibleEulerEquations3D <: AbstractCompressibleEulerEquations{3, 5}
  gamma::Float64
end

function CompressibleEulerEquations3D()
  gamma = parameter("gamma", 1.4)

  CompressibleEulerEquations3D(gamma)
end


get_name(::CompressibleEulerEquations3D) = "CompressibleEulerEquations3D"
varnames_cons(::CompressibleEulerEquations3D) = @SVector ["rho", "rho_v1", "rho_v2", "rho_v3", "rho_e"]
varnames_prim(::CompressibleEulerEquations3D) = @SVector ["rho", "v1", "v2", "v3", "p"]


# Set initial conditions at physical location `x` for time `t`
function initial_condition_density_pulse(x, t, equation::CompressibleEulerEquations3D)
  rho = 1 + exp(-(x[1]^2 + x[2]^2 + x[3]^2))/2
  v1 = 1
  v2 = 1
  v3 = 1
  rho_v1 = rho * v1
  rho_v2 = rho * v2
  rho_v3 = rho * v3
  p = 1
  rho_e = p/(equation.gamma - 1) + 1/2 * rho * (v1^2 + v2^2 + v3^2)
  return @SVector [rho, rho_v1, rho_v2, rho_v3, rho_e]
end

function initial_condition_constant(x, t, equation::CompressibleEulerEquations3D)
  rho = 1.0
  rho_v1 = 0.1
  rho_v2 = -0.2
  rho_v3 = 0.7
  rho_e = 10.0
  return @SVector [rho, rho_v1, rho_v2, rho_v3, rho_e]
end

function initial_condition_convergence_test(x, t, equation::CompressibleEulerEquations3D)
  c = 2
  A = 0.1
  L = 2
  f = 1/L
  ω = 2 * pi * f
  ini = c + A * sin(ω * (x[1] + x[2] + x[3] - t))

  rho = ini
  rho_v1 = ini
  rho_v2 = ini
  rho_v3 = ini
  rho_e = ini^2

  return @SVector [rho, rho_v1, rho_v2, rho_v3, rho_e]
end

function initial_condition_weak_blast_wave(x, t, equation::CompressibleEulerEquations3D)
  # From Hennemann & Gassner JCP paper 2020 (Sec. 6.3)
  # Set up spherical coordinates
  inicenter = (0, 0, 0)
  x_norm = x[1] - inicenter[1]
  y_norm = x[2] - inicenter[2]
  z_norm = x[3] - inicenter[3]
  r = sqrt(x_norm^2 + y_norm^2 + z_norm^2)
  phi   = atan(y_norm, x_norm)
  theta = iszero(r) ? 0.0 : acos(z_norm / r)

  # Calculate primitive variables
  rho = r > 0.5 ? 1.0 : 1.1691
  v1  = r > 0.5 ? 0.0 : 0.1882 * cos(phi) * sin(theta)
  v2  = r > 0.5 ? 0.0 : 0.1882 * sin(phi) * sin(theta)
  v3  = r > 0.5 ? 0.0 : 0.1882 * cos(theta)
  p   = r > 0.5 ? 1.0 : 1.245

  return prim2cons(SVector(rho, v1, v2, v3, p), equation)
end

function initial_condition_sedov_blast_wave(x, t, equation::CompressibleEulerEquations3D)
  # Calculate radius as distance from origin
  r = sqrt(x[1]^2 + x[2]^2 + x[3]^2)

  # Setup based on http://flash.uchicago.edu/site/flashcode/user_support/flash4_ug_4p62/node184.html#SECTION010114000000000000000
  r0 = 0.25 # = 4.0 * smallest dx (for domain length=8 and max-ref=7)
  E = 1.0
  p_inner   = (equation.gamma - 1) * E / (4/3 * pi * r0^3)
  p_ambient = 1e-5 # = true Sedov setup

  # Calculate primitive variables
  rho = 1.0
  v1 = 0.0
  v2 = 0.0
  v3 = 0.0

  # use a logistic function to tranfer pressure value smoothly
  k  = -50.0 # sharpness of transfer
  logistic_function_p = p_inner/(1.0 + exp(-k*(r - r0)))
  p = max(logistic_function_p, p_ambient)

  return prim2cons(SVector(rho, v1, v2, v3, p), equation)
end

function initial_condition_eoc_test_coupled_euler_gravity(x, t, equation::CompressibleEulerEquations3D)
  # OBS! this assumes that γ = 2 other manufactured source terms are incorrect
  if equation.gamma != 2.0
    error("adiabatic constant must be 2 for the coupling convergence test")
  end
  c = 2.0
  A = 0.1
  ini = c + A * sin(pi * (x[1] + x[2] + x[3] - t))
  G = 1.0 # gravitational constant

  rho = ini
  v1 = 1.0
  v2 = 1.0
  v3 = 1.0
  p = ini^2 * G * 2 / (3 * pi) # "3" is the number of spatial dimensions

  return prim2cons(SVector(rho, v1, v2, v3, p), equation)
end

function initial_condition_sedov_self_gravity(x, t, equation::CompressibleEulerEquations3D)
  # Calculate radius as distance from origin
  r = sqrt(x[1]^2 + x[2]^2 + x[3]^2)

  # Setup based on http://flash.uchicago.edu/site/flashcode/user_support/flash4_ug_4p62/node184.html#SECTION010114000000000000000
  r0 = 0.25 # = 4.0 * smallest dx (for domain length=8 and max-ref=7)
  E = 1.0
  p_inner   = (equation.gamma - 1) * E / (4/3 * pi * r0^3)
  p_ambient = 1e-5 # = true Sedov setup

  # Calculate primitive variables
  # use a logistic function to tranfer density value smoothly
  L  = 1.0    # maximum of function
  x0 = 1.0    # center point of function
  k  = -50.0 # sharpness of transfer
  logistic_function_rho = L/(1.0 + exp(-k*(r - x0)))
  rho_ambient = 1e-5
  rho = max(logistic_function_rho, rho_ambient) # clip background density to not be so tiny

  # velocities are zero
  v1 = 0.0
  v2 = 0.0
  v3 = 0.0

  # use a logistic function to tranfer pressure value smoothly
  logistic_function_p = p_inner/(1.0 + exp(-k*(r - r0)))
  p = max(logistic_function_p, p_ambient)

  return prim2cons(SVector(rho, v1, v2, v3, p), equation)
end

function initial_condition_taylor_green_vortex(x, t, equation::CompressibleEulerEquations3D)
  A  = 1.0 # magnitude of speed
  Ms = 0.1 # maximum Mach number

  rho = 1.0
  v1  =  A * sin(x[1]) * cos(x[2]) * cos(x[3])
  v2  = -A * cos(x[1]) * sin(x[2]) * cos(x[3])
  v3  = 0.0
  p   = (A / Ms)^2 * rho / equation.gamma # scaling to get Ms
  p   = p + 1.0/16.0 * A^2 * rho * (cos(2*x[1])*cos(2*x[3]) + 2*cos(2*x[2]) + 2*cos(2*x[1]) + cos(2*x[2])*cos(2*x[3]))

  return prim2cons(SVector(rho, v1, v2, v3, p), equation)
end

function initial_condition_blob(x, t, equation::CompressibleEulerEquations3D)
  # blob test case, see Agertz et al. https://arxiv.org/pdf/astro-ph/0610051.pdf
  # other reference: https://arxiv.org/pdf/astro-ph/0610051.pdf
  # change discontinuity to tanh
  # typical domain is rectangular, we change it to a square, as Trixi can only do squares
  # resolution 128^3, 256^3
  # domain size is [-20.0,20.0]^3
  # gamma = 5/3 for this test case
  R = 1.0 # radius of the blob
  # background density
  rho = 1.0
  Chi = 10.0 # density contrast
  # reference time of characteristic growth of KH instability equal to 1.0
  tau_kh = 1.0
  tau_cr = tau_kh / 1.6 # crushing time
  # determine background velocity
  v1 = 2 * R * sqrt(Chi) / tau_cr
  v2 = 0.0
  v3 = 0.0
  Ma0 = 2.7 # background flow Mach number Ma=v/c
  c = v1 / Ma0 # sound speed
  # use perfect gas assumption to compute background pressure via the sound speed c^2 = gamma * pressure/density
  p = c * c * rho / equation.gamma
  # initial center of the blob
  inicenter = [-15, 0, 0]
  x_rel = x - inicenter
  r = sqrt(x_rel[1]^2 + x_rel[2]^2 + x_rel[3]^2)
  # steepness of the tanh transition zone
  slope = 2
  # density blob
  rho = rho + (Chi - 1) * 0.5 * (1 + (tanh(slope * (r + R)) - (tanh(slope *(r - R)) + 1)))
  # velocity blob is zero
  v1 = v1 - v1 * 0.5 * (1 + (tanh(slope *(r + R)) - (tanh(slope *(r - R)) + 1)))
  return prim2cons(SVector(rho, v1, v2, v3, p), equation)
end



# Apply boundary conditions
function boundary_condition_sedov_self_gravity(u_inner, orientation, direction, x, t,
                                                surface_flux_function,
                                                equation::CompressibleEulerEquations3D)
  # velocities are zero, density/pressure are ambient values according to
  # initial_condition_sedov_self_gravity
  rho = 1e-5
  v1 = 0.0
  v2 = 0.0
  v3 = 0.0
  p = 1e-5

  u_boundary = prim2cons(SVector(rho, v1, v2, v3, p), equation)

  # Calculate boundary flux
  if direction in (2, 4, 6) # u_inner is "left" of boundary, u_boundary is "right" of boundary
    flux = surface_flux_function(u_inner, u_boundary, orientation, equation)
  else # u_boundary is "left" of boundary, u_inner is "right" of boundary
    flux = surface_flux_function(u_boundary, u_inner, orientation, equation)
  end

  return flux
end


# Apply source terms
function source_terms_convergence_test(ut, u, x, element_id, t, n_nodes, equation::CompressibleEulerEquations3D)
  # Same settings as in `initial_condition`
  c = 2
  A = 0.1
  L = 2
  f = 1/L
  ω = 2 * pi * f
  γ = equation.gamma

  for k in 1:n_nodes, j in 1:n_nodes, i in 1:n_nodes
    x1 = x[1, i, j, k, element_id]
    x2 = x[2, i, j, k, element_id]
    x3 = x[3, i, j, k, element_id]

    si, co = sincos(((x1 + x2 + x3) - t) * ω)
    tmp1 = si * A
    tmp2 = co * A * ω
    tmp3 = ((((((4 * tmp1 * γ - 4 * tmp1) + 4 * c * γ) - 4c) - 3γ) + 7) * tmp2) / 2

    ut[1, i, j, k, element_id] += 2 * tmp2
    ut[2, i, j, k, element_id] += tmp3
    ut[3, i, j, k, element_id] += tmp3
    ut[4, i, j, k, element_id] += tmp3
    ut[5, i, j, k, element_id] += ((((((12 * tmp1 * γ - 4 * tmp1) + 12 * c * γ) - 4c) - 9γ) + 9) * tmp2) / 2

    # Original terms (without performanc enhancements)
    # tmp2 = ((((((4 * sin(((x1 + x2 + x3) - t) * ω) * A * γ - 4 * sin(((x1 + x2 + x3) - t) * ω) * A) + 4 * c * γ) - 4c) - 3γ) + 7) * cos(((x1 + x2 + x3) - t) * ω) * A * ω) / 2
    # ut[1, i, j, k, element_id] += 2 * cos(((x1 + x2 + x3) - t) * ω) * A * ω
    # ut[2, i, j, k, element_id] += tmp2
    # ut[3, i, j, k, element_id] += tmp2
    # ut[4, i, j, k, element_id] += tmp2
    # ut[5, i, j, k, element_id] += ((((((12 * sin(((x1 + x2 + x3) - t) * ω) * A * γ - 4 * sin(((x1 + x2 + x3) - t) * ω) * A) + 12 * c * γ) - 4c) - 9γ) + 9) * cos(((x1 + x2 + x3) - t) * ω) * A * ω) / 2
  end

  return nothing
end

function source_terms_eoc_test_coupled_euler_gravity(ut, u, x, element_id, t, n_nodes, equation::CompressibleEulerEquations3D)
  # Same settings as in `initial_condition_eoc_test_coupled_euler_gravity`
  c = 2.0
  A = 0.1
  G = 1.0 # gravitational constant, must match coupling solver
  C_grav = -4 * G / (3 * pi) # "3" is the number of spatial dimensions  # 2D: -2.0*G/pi

  for k in 1:n_nodes, j in 1:n_nodes, i in 1:n_nodes
    x1 = x[1, i, j, k, element_id]
    x2 = x[2, i, j, k, element_id]
    x3 = x[3, i, j, k, element_id]
    rhox = A * pi * cos(pi * (x1 + x2 + x3 - t))
    rho  = c + A * sin(pi * (x1 + x2 + x3 - t))

    # In "2 * rhox", the "2" is "number of spatial dimensions minus one"
    ut[1, i, j, k, element_id] += 2 * rhox
    ut[2, i, j, k, element_id] += 2 * rhox
    ut[3, i, j, k, element_id] += 2 * rhox
    ut[4, i, j, k, element_id] += 2 * rhox
    ut[5, i, j, k, element_id] += 2 * rhox * (3/2 - C_grav*rho) # "3" in "3/2" is the number of spatial dimensions
  end

  return nothing
end

function source_terms_eoc_test_euler(ut, u, x, element_id, t, n_nodes, equation::CompressibleEulerEquations3D)
  # Same settings as in `initial_condition_eoc_test_coupled_euler_gravity`
  c = 2.0
  A = 0.1
  G = 1.0
  C_grav = -4 * G / (3 * pi) # "3" is the number of spatial dimensions

  for k in 1:n_nodes, j in 1:n_nodes, i in 1:n_nodes
    x1 = x[1, i, j, k, element_id]
    x2 = x[2, i, j, k, element_id]
    x3 = x[3, i, j, k, element_id]
    si, co = sincos(pi * (x1 + x2 + x3 - t))
    rhox = A * pi * co
    rho  = c + A *  si

    ut[1, i, j, k, element_id] += rhox *  2
    ut[2, i, j, k, element_id] += rhox * (2 -     C_grav * rho)
    ut[3, i, j, k, element_id] += rhox * (2 -     C_grav * rho)
    ut[4, i, j, k, element_id] += rhox * (2 -     C_grav * rho)
    ut[5, i, j, k, element_id] += rhox * (3 - 5 * C_grav * rho)
  end

  return nothing
end

# Empty source terms required for coupled Euler-gravity simulations
function source_terms_harmonic(ut, u, x, element_id, t, n_nodes, equation::CompressibleEulerEquations3D)
  # OBS! used for self-gravitating Sedov blast
  # TODO: make this cleaner and let each solver have a different source term name
  return nothing
end

# Calculate 1D flux for a single point
@inline function calcflux(u, orientation, equation::CompressibleEulerEquations3D)
  rho, rho_v1, rho_v2, rho_v3, rho_e = u
  v1 = rho_v1/rho
  v2 = rho_v2/rho
  v3 = rho_v3/rho
  p = (equation.gamma - 1) * (rho_e - 1/2 * rho * (v1^2 + v2^2 + v3^2))
  if orientation == 1
    f1 = rho_v1
    f2 = rho_v1 * v1 + p
    f3 = rho_v1 * v2
    f4 = rho_v1 * v3
    f5 = (rho_e + p) * v1
  elseif orientation == 2
    f1 = rho_v2
    f2 = rho_v2 * v1
    f3 = rho_v2 * v2 + p
    f4 = rho_v2 * v3
    f5 = (rho_e + p) * v2
  else
    f1 = rho_v3
    f2 = rho_v3 * v1
    f3 = rho_v3 * v2
    f4 = rho_v3 * v3 + p
    f5 = (rho_e + p) * v3
  end
  return SVector(f1, f2, f3, f4, f5)
end


"""
    function flux_shima_etal(u_ll, u_rr, orientation, equation::CompressibleEulerEquations3D)

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
@inline function flux_shima_etal(u_ll, u_rr, orientation, equation::CompressibleEulerEquations3D)
  # Unpack left and right state
  rho_ll, rho_v1_ll, rho_v2_ll, rho_v3_ll, rho_e_ll = u_ll
  rho_rr, rho_v1_rr, rho_v2_rr, rho_v3_rr, rho_e_rr = u_rr

  v1_ll = rho_v1_ll / rho_ll
  v2_ll = rho_v2_ll / rho_ll
  v3_ll = rho_v3_ll / rho_ll
  v1_rr = rho_v1_rr / rho_rr
  v2_rr = rho_v2_rr / rho_rr
  v3_rr = rho_v3_rr / rho_rr
  p_ll =  (equation.gamma - 1) * (rho_e_ll - 1/2 * rho_ll * (v1_ll^2 + v2_ll^2 + v3_ll^2))
  p_rr =  (equation.gamma - 1) * (rho_e_rr - 1/2 * rho_rr * (v1_rr^2 + v2_rr^2 + v3_rr^2))

  # Average each factor of products in flux
  rho_avg = 1/2 * (rho_ll + rho_rr)
  v1_avg  = 1/2 * ( v1_ll +  v1_rr)
  v2_avg  = 1/2 * ( v2_ll +  v2_rr)
  v3_avg  = 1/2 * ( v3_ll +  v3_rr)
  p_avg   = 1/2 * (  p_ll +   p_rr)
  kin_avg = 1/2 * (v1_ll*v1_rr + v2_ll*v2_rr + v3_ll*v3_rr)

  # Calculate fluxes depending on orientation
  if orientation == 1
    pv1_avg = 1/2 * (p_ll*v1_rr + p_rr*v1_ll)
    f1 = rho_avg * v1_avg
    f2 = rho_avg * v1_avg * v1_avg + p_avg
    f3 = rho_avg * v1_avg * v2_avg
    f4 = rho_avg * v1_avg * v3_avg
    f5 = p_avg*v1_avg/(equation.gamma-1) + rho_avg*v1_avg*kin_avg + pv1_avg
  elseif orientation == 2
    pv2_avg = 1/2 * (p_ll*v2_rr + p_rr*v2_ll)
    f1 = rho_avg * v2_avg
    f2 = rho_avg * v2_avg * v1_avg
    f3 = rho_avg * v2_avg * v2_avg + p_avg
    f4 = rho_avg * v2_avg * v3_avg
    f5 = p_avg*v2_avg/(equation.gamma-1) + rho_avg*v2_avg*kin_avg + pv2_avg
  else
    pv3_avg = 1/2 * (p_ll*v3_rr + p_rr*v3_ll)
    f1 = rho_avg * v3_avg
    f2 = rho_avg * v3_avg * v1_avg
    f3 = rho_avg * v3_avg * v2_avg
    f4 = rho_avg * v3_avg * v3_avg + p_avg
    f5 = p_avg*v3_avg/(equation.gamma-1) + rho_avg*v3_avg*kin_avg + pv3_avg
  end

  return SVector(f1, f2, f3, f4, f5)
end


"""
    flux_kennedy_gruber(u_ll, u_rr, orientation, equation::CompressibleEulerEquations3D)

Kinetic energy preserving two-point flux by Kennedy and Gruber (2008)
  Reduced aliasing formulations of the convective terms within the
  Navier-Stokes equations for a compressible fluid
[DOI: 10.1016/j.jcp.2007.09.020](https://doi.org/10.1016/j.jcp.2007.09.020)
"""
@inline function flux_kennedy_gruber(u_ll, u_rr, orientation, equation::CompressibleEulerEquations3D)
  # Unpack left and right state
  rho_ll, rho_v1_ll, rho_v2_ll, rho_v3_ll, rho_e_ll = u_ll
  rho_rr, rho_v1_rr, rho_v2_rr, rho_v3_rr, rho_e_rr = u_rr

  v1_ll = rho_v1_ll/rho_ll
  v2_ll = rho_v2_ll/rho_ll
  v3_ll = rho_v3_ll/rho_ll
  v1_rr = rho_v1_rr/rho_rr
  v2_rr = rho_v2_rr/rho_rr
  v3_rr = rho_v3_rr/rho_rr

  # Average each factor of products in flux
  rho_avg = 1/2 * (rho_ll + rho_rr)
  v1_avg = 1/2 * (v1_ll + v1_rr)
  v2_avg = 1/2 * (v2_ll + v2_rr)
  v3_avg = 1/2 * (v3_ll + v3_rr)
  p_avg = 1/2 * ((equation.gamma - 1) * (rho_e_ll - 1/2 * rho_ll * (v1_ll^2 + v2_ll^2 + v3_ll^2)) +
                 (equation.gamma - 1) * (rho_e_rr - 1/2 * rho_rr * (v1_rr^2 + v2_rr^2 + v3_rr^2)))
  e_avg = 1/2 * (rho_e_ll/rho_ll + rho_e_rr/rho_rr)

  # Calculate fluxes depending on orientation
  if orientation == 1
    f1 = rho_avg * v1_avg
    f2 = rho_avg * v1_avg * v1_avg + p_avg
    f3 = rho_avg * v1_avg * v2_avg
    f4 = rho_avg * v1_avg * v3_avg
    f5 = (rho_avg * e_avg + p_avg) * v1_avg
  elseif orientation == 2
    f1 = rho_avg * v2_avg
    f2 = rho_avg * v2_avg * v1_avg
    f3 = rho_avg * v2_avg * v2_avg + p_avg
    f4 = rho_avg * v2_avg * v3_avg
    f5 = (rho_avg * e_avg + p_avg) * v2_avg
  else
    f1 = rho_avg * v3_avg
    f2 = rho_avg * v3_avg * v1_avg
    f3 = rho_avg * v3_avg * v2_avg
    f4 = rho_avg * v3_avg * v3_avg + p_avg
    f5 = (rho_avg * e_avg + p_avg) * v3_avg
  end

  return SVector(f1, f2, f3, f4, f5)
end


"""
    flux_chandrashekar(u_ll, u_rr, orientation, equation::CompressibleEulerEquations3D)

Entropy conserving two-point flux by Chandrashekar (2013)
  Kinetic Energy Preserving and Entropy Stable Finite Volume Schemes
  for Compressible Euler and Navier-Stokes Equations
[DOI: 10.4208/cicp.170712.010313a](https://doi.org/10.4208/cicp.170712.010313a)
"""
@inline function flux_chandrashekar(u_ll, u_rr, orientation, equation::CompressibleEulerEquations3D)
  # Unpack left and right state
  rho_ll, rho_v1_ll, rho_v2_ll, rho_v3_ll, rho_e_ll = u_ll
  rho_rr, rho_v1_rr, rho_v2_rr, rho_v3_rr, rho_e_rr = u_rr

  v1_ll = rho_v1_ll/rho_ll
  v2_ll = rho_v2_ll/rho_ll
  v3_ll = rho_v3_ll/rho_ll
  v1_rr = rho_v1_rr/rho_rr
  v2_rr = rho_v2_rr/rho_rr
  v3_rr = rho_v3_rr/rho_rr
  p_ll =  (equation.gamma - 1) * (rho_e_ll - 1/2 * rho_ll * (v1_ll^2 + v2_ll^2 + v3_ll^2))
  p_rr =  (equation.gamma - 1) * (rho_e_rr - 1/2 * rho_rr * (v1_rr^2 + v2_rr^2 + v3_rr^2))
  beta_ll = 0.5*rho_ll/p_ll
  beta_rr = 0.5*rho_rr/p_rr
  specific_kin_ll = 0.5*(v1_ll^2 + v2_ll^2 + v3_ll^2)
  specific_kin_rr = 0.5*(v1_rr^2 + v2_rr^2 + v3_rr^2)

  # Compute the necessary mean values
  rho_avg  = 0.5*(rho_ll+rho_rr)
  rho_mean = ln_mean(rho_ll,rho_rr)
  beta_mean = ln_mean(beta_ll,beta_rr)
  beta_avg = 0.5*(beta_ll+beta_rr)
  v1_avg = 0.5*(v1_ll+v1_rr)
  v2_avg = 0.5*(v2_ll+v2_rr)
  v3_avg = 0.5*(v3_ll+v3_rr)
  p_mean = 0.5*rho_avg/beta_avg
  velocity_square_avg = specific_kin_ll + specific_kin_rr

  # Calculate fluxes depending on orientation
  if orientation == 1
    f1 = rho_mean * v1_avg
    f2 = f1 * v1_avg + p_mean
    f3 = f1 * v2_avg
    f4 = f1 * v3_avg
    f5 = f1 * 0.5*(1/(equation.gamma-1)/beta_mean - velocity_square_avg)+ f2*v1_avg + f3*v2_avg + f4*v3_avg
  elseif orientation == 2
    f1 = rho_mean * v2_avg
    f2 = f1 * v1_avg
    f3 = f1 * v2_avg + p_mean
    f4 = f1 * v3_avg
    f5 = f1 * 0.5*(1/(equation.gamma-1)/beta_mean - velocity_square_avg)+ f2*v1_avg + f3*v2_avg + f4*v3_avg
  else
    f1 = rho_mean * v3_avg
    f2 = f1 * v1_avg
    f3 = f1 * v2_avg
    f4 = f1 * v3_avg + p_mean
    f5 = f1 * 0.5*(1/(equation.gamma-1)/beta_mean - velocity_square_avg)+ f2*v1_avg + f3*v2_avg + f4*v3_avg
  end

  return SVector(f1, f2, f3, f4, f5)
end


"""
    flux_ranocha(u_ll, u_rr, orientation, equation::CompressibleEulerEquations3D)

Entropy conserving and kinetic energy preserving two-point flux by Ranocha (2018)
  Generalised Summation-by-Parts Operators and Entropy Stability of Numerical Methods
  for Hyperbolic Balance Laws
[PhD thesis, TU Braunschweig](https://cuvillier.de/en/shop/publications/7743)
See also Ranocha (2020)
  Entropy Conserving and Kinetic Energy Preserving Numerical Methods for
  the Euler Equations Using Summation-by-Parts Operators
[Proceedings of ICOSAHOM 2018](https://doi.org/10.1007/978-3-030-39647-3_42)
"""
@inline function flux_ranocha(u_ll, u_rr, orientation, equation::CompressibleEulerEquations3D)
  # Unpack left and right state
  rho_ll, rho_v1_ll, rho_v2_ll, rho_v3_ll, rho_e_ll = u_ll
  rho_rr, rho_v1_rr, rho_v2_rr, rho_v3_rr, rho_e_rr = u_rr

  v1_ll = rho_v1_ll / rho_ll
  v2_ll = rho_v2_ll / rho_ll
  v3_ll = rho_v3_ll / rho_ll
  v1_rr = rho_v1_rr / rho_rr
  v2_rr = rho_v2_rr / rho_rr
  v3_rr = rho_v3_rr / rho_rr
  p_ll =  (equation.gamma - 1) * (rho_e_ll - 0.5 * rho_ll * (v1_ll^2 + v2_ll^2 + v3_ll^2))
  p_rr =  (equation.gamma - 1) * (rho_e_rr - 0.5 * rho_rr * (v1_rr^2 + v2_rr^2 + v3_rr^2))

  # Compute the necessary mean values
  rho_mean   = ln_mean(rho_ll, rho_rr)
  rho_p_mean = ln_mean(rho_ll / p_ll, rho_rr / p_rr)
  v1_avg = 0.5 * (v1_ll + v1_rr)
  v2_avg = 0.5 * (v2_ll + v2_rr)
  v3_avg = 0.5 * (v3_ll + v3_rr)
  p_avg  = 0.5 * (p_ll + p_rr)
  velocity_square_avg = 0.5 * (v1_ll*v1_rr + v2_ll*v2_rr + v3_ll*v3_rr)

  # Calculate fluxes depending on orientation
  if orientation == 1
    f1 = rho_mean * v1_avg
    f2 = f1 * v1_avg + p_avg
    f3 = f1 * v2_avg
    f4 = f1 * v3_avg
    f5 = f1 * ( velocity_square_avg + 1 / ((equation.gamma-1) * rho_p_mean) ) + 0.5 * (p_ll*v1_rr + p_rr*v1_ll)
  elseif orientation == 2
    f1 = rho_mean * v2_avg
    f2 = f1 * v1_avg
    f3 = f1 * v2_avg + p_avg
    f4 = f1 * v3_avg
    f5 = f1 * ( velocity_square_avg + 1 / ((equation.gamma-1) * rho_p_mean) ) + 0.5 * (p_ll*v2_rr + p_rr*v2_ll)
  else # orientation == 3
    f1 = rho_mean * v3_avg
    f2 = f1 * v1_avg
    f3 = f1 * v2_avg
    f4 = f1 * v3_avg + p_avg
    f5 = f1 * ( velocity_square_avg + 1 / ((equation.gamma-1) * rho_p_mean) ) + 0.5 * (p_ll*v3_rr + p_rr*v3_ll)
  end

  return SVector(f1, f2, f3, f4, f5)
end


function flux_lax_friedrichs(u_ll, u_rr, orientation, equation::CompressibleEulerEquations3D)
  # Calculate primitive variables and speed of sound
  rho_ll, rho_v1_ll, rho_v2_ll, rho_v3_ll, rho_e_ll = u_ll
  rho_rr, rho_v1_rr, rho_v2_rr, rho_v3_rr, rho_e_rr = u_rr

  v1_ll = rho_v1_ll / rho_ll
  v2_ll = rho_v2_ll / rho_ll
  v3_ll = rho_v3_ll / rho_ll
  v_mag_ll = sqrt(v1_ll^2 + v2_ll^2 + v3_ll^2)
  p_ll = (equation.gamma - 1) * (rho_e_ll - 1/2 * rho_ll * v_mag_ll^2)
  c_ll = sqrt(equation.gamma * p_ll / rho_ll)
  v1_rr = rho_v1_rr / rho_rr
  v2_rr = rho_v2_rr / rho_rr
  v3_rr = rho_v3_rr / rho_rr
  v_mag_rr = sqrt(v1_rr^2 + v2_rr^2 + v3_rr^2)
  p_rr = (equation.gamma - 1) * (rho_e_rr - 1/2 * rho_rr * v_mag_rr^2)
  c_rr = sqrt(equation.gamma * p_rr / rho_rr)

  # Obtain left and right fluxes
  f_ll = calcflux(u_ll, orientation, equation)
  f_rr = calcflux(u_rr, orientation, equation)

  λ_max = max(v_mag_ll, v_mag_rr) + max(c_ll, c_rr)
  f1 = 1/2 * (f_ll[1] + f_rr[1]) - 1/2 * λ_max * (rho_rr    - rho_ll)
  f2 = 1/2 * (f_ll[2] + f_rr[2]) - 1/2 * λ_max * (rho_v1_rr - rho_v1_ll)
  f3 = 1/2 * (f_ll[3] + f_rr[3]) - 1/2 * λ_max * (rho_v2_rr - rho_v2_ll)
  f4 = 1/2 * (f_ll[4] + f_rr[4]) - 1/2 * λ_max * (rho_v3_rr - rho_v3_ll)
  f5 = 1/2 * (f_ll[5] + f_rr[5]) - 1/2 * λ_max * (rho_e_rr  - rho_e_ll)

  return SVector(f1, f2, f3, f4, f5)
end


function flux_hll(u_ll, u_rr, orientation, equation::CompressibleEulerEquations3D)
  # Calculate primitive variables and speed of sound
  rho_ll, rho_v1_ll, rho_v2_ll, rho_v3_ll, rho_e_ll = u_ll
  rho_rr, rho_v1_rr, rho_v2_rr, rho_v3_rr, rho_e_rr = u_rr

  v1_ll = rho_v1_ll / rho_ll
  v2_ll = rho_v2_ll / rho_ll
  v3_ll = rho_v3_ll / rho_ll
  p_ll = (equation.gamma - 1) * (rho_e_ll - 1/2 * rho_ll * (v1_ll^2 + v2_ll^2 + v3_ll^2))

  v1_rr = rho_v1_rr / rho_rr
  v2_rr = rho_v2_rr / rho_rr
  v3_rr = rho_v3_rr / rho_rr
  p_rr = (equation.gamma - 1) * (rho_e_rr - 1/2 * rho_rr * (v1_rr^2 + v2_rr^2 + v3_rr^2))

  # Obtain left and right fluxes
  f_ll = calcflux(u_ll, orientation, equation)
  f_rr = calcflux(u_rr, orientation, equation)

  if orientation == 1 # x-direction
    Ssl = v1_ll - sqrt(equation.gamma * p_ll / rho_ll)
    Ssr = v1_rr + sqrt(equation.gamma * p_rr / rho_rr)
  elseif orientation == 2 # y-direction
    Ssl = v2_ll - sqrt(equation.gamma * p_ll / rho_ll)
    Ssr = v2_rr + sqrt(equation.gamma * p_rr / rho_rr)
  else # z-direction
    Ssl = v3_ll - sqrt(equation.gamma * p_ll / rho_ll)
    Ssr = v3_rr + sqrt(equation.gamma * p_rr / rho_rr)
  end

  if Ssl >= 0.0 && Ssr > 0.0
    f1 = f_ll[1]
    f2 = f_ll[2]
    f3 = f_ll[3]
    f4 = f_ll[4]
    f5 = f_ll[5]
  elseif Ssr <= 0.0 && Ssl < 0.0
    f1 = f_rr[1]
    f2 = f_rr[2]
    f3 = f_rr[3]
    f4 = f_rr[4]
    f5 = f_rr[5]
  else
    f1 = (Ssr*f_ll[1] - Ssl*f_rr[1] + Ssl*Ssr*(rho_rr    - rho_ll))    / (Ssr - Ssl)
    f2 = (Ssr*f_ll[2] - Ssl*f_rr[2] + Ssl*Ssr*(rho_v1_rr - rho_v1_ll)) / (Ssr - Ssl)
    f3 = (Ssr*f_ll[3] - Ssl*f_rr[3] + Ssl*Ssr*(rho_v2_rr - rho_v2_ll)) / (Ssr - Ssl)
    f4 = (Ssr*f_ll[4] - Ssl*f_rr[4] + Ssl*Ssr*(rho_v3_rr - rho_v3_ll)) / (Ssr - Ssl)
    f5 = (Ssr*f_ll[5] - Ssl*f_rr[5] + Ssl*Ssr*(rho_e_rr  - rho_e_ll))  / (Ssr - Ssl)
  end

  return SVector(f1, f2, f3, f4, f5)
end


 """
    flux_hllc(u_ll, u_rr, orientation, equation::CompressibleEulerEquations3D)

Computes the HLLC flux (HLL with Contact) for compressible Euler equations developed by E.F. Toro
[Lecture slides](http://www.prague-sum.com/download/2012/Toro_2-HLLC-RiemannSolver.pdf)
Signal speeds: [DOI: 10.1137/S1064827593260140](https://doi.org/10.1137/S1064827593260140)
"""
function flux_hllc(u_ll, u_rr, orientation, equation::CompressibleEulerEquations3D)
  # Calculate primitive variables and speed of sound
  rho_ll, rho_v1_ll, rho_v2_ll, rho_v3_ll, rho_e_ll = u_ll
  rho_rr, rho_v1_rr, rho_v2_rr, rho_v3_rr, rho_e_rr = u_rr

  v1_ll = rho_v1_ll / rho_ll
  v2_ll = rho_v2_ll / rho_ll
  v3_ll = rho_v3_ll / rho_ll
  e_ll  = rho_e_ll / rho_ll
  p_ll = (equation.gamma - 1) * (rho_e_ll - 1/2 * rho_ll * (v1_ll^2 + v2_ll^2 + v3_ll^2))
  c_ll = sqrt(equation.gamma*p_ll/rho_ll)

  v1_rr = rho_v1_rr / rho_rr
  v2_rr = rho_v2_rr / rho_rr
  v3_rr = rho_v3_rr / rho_rr
  e_rr  = rho_e_rr / rho_rr
  p_rr = (equation.gamma - 1) * (rho_e_rr - 1/2 * rho_rr * (v1_rr^2 + v2_rr^2 + v3_rr^2))
  c_rr = sqrt(equation.gamma*p_rr/rho_rr)

  # Obtain left and right fluxes
  f_ll = calcflux(u_ll, orientation, equation)
  f_rr = calcflux(u_rr, orientation, equation)


  # Compute Roe averages
  sqrt_rho_ll = sqrt(rho_ll)
  sqrt_rho_rr = sqrt(rho_rr)
  sum_sqrt_rho = sqrt_rho_ll + sqrt_rho_rr
  if orientation == 1 # x-direction
    vel_L = v1_ll
    vel_R = v1_rr
    ekin_roe = (sqrt_rho_ll * v2_ll + sqrt_rho_rr * v2_rr)^2 + (sqrt_rho_ll * v3_ll + sqrt_rho_rr * v3_rr)^2
  elseif orientation == 2 # y-direction
    vel_L = v2_ll
    vel_R = v2_rr
    ekin_roe = (sqrt_rho_ll * v1_ll + sqrt_rho_rr * v1_rr)^2 + (sqrt_rho_ll * v3_ll + sqrt_rho_rr * v3_rr)^2
  else # z-direction
    vel_L = v3_ll
    vel_R = v3_rr
    ekin_roe = (sqrt_rho_ll * v1_ll + sqrt_rho_rr * v1_rr)^2 + (sqrt_rho_ll * v2_ll + sqrt_rho_rr * v2_rr)^2
  end
  vel_roe = (sqrt_rho_ll * vel_L + sqrt_rho_rr * vel_R) / sum_sqrt_rho
  ekin_roe = 0.5 * (vel_roe^2 + ekin_roe / sum_sqrt_rho^2)
  H_ll = (rho_e_ll + p_ll) / rho_ll
  H_rr = (rho_e_rr + p_rr) / rho_rr
  H_roe = (sqrt_rho_ll * H_ll + sqrt_rho_rr * H_rr) / sum_sqrt_rho
  c_roe = sqrt((equation.gamma - 1) * (H_roe - ekin_roe))
  Ssl = min(vel_L - c_ll, vel_roe - c_roe)
  Ssr = max(vel_R + c_rr, vel_roe + c_roe)
  sMu_L = Ssl - vel_L
  sMu_R = Ssr - vel_R

  if Ssl >= 0.0
    f1 = f_ll[1]
    f2 = f_ll[2]
    f3 = f_ll[3]
    f4 = f_ll[4]
    f5 = f_ll[5]
  elseif Ssr <= 0.0
    f1 = f_rr[1]
    f2 = f_rr[2]
    f3 = f_rr[3]
    f4 = f_rr[4]
    f5 = f_rr[5]
  else
    SStar = (p_rr - p_ll + rho_ll*vel_L*sMu_L - rho_rr*vel_R*sMu_R) / (rho_ll*sMu_L - rho_rr*sMu_R)
    if Ssl <= 0.0 <= SStar
      densStar = rho_ll*sMu_L / (Ssl-SStar)
      enerStar = e_ll + (SStar - vel_L) * (SStar + p_ll / (rho_ll * sMu_L))
      UStar1 = densStar
      UStar5 = densStar*enerStar
      if orientation == 1 # x-direction
        UStar2 = densStar*SStar
        UStar3 = densStar*v2_ll
        UStar4 = densStar*v3_ll
      elseif orientation == 2 # y-direction
        UStar2 = densStar*v1_ll
        UStar3 = densStar*SStar
        UStar4 = densStar*v3_ll
      else # z-direction
        UStar2 = densStar*v1_ll
        UStar3 = densStar*v2_ll
        UStar4 = densStar*SStar
      end
      f1 = f_ll[1]+Ssl*(UStar1 - rho_ll)
      f2 = f_ll[2]+Ssl*(UStar2 - rho_v1_ll)
      f3 = f_ll[3]+Ssl*(UStar3 - rho_v2_ll)
      f4 = f_ll[4]+Ssl*(UStar4 - rho_v3_ll)
      f5 = f_ll[5]+Ssl*(UStar5 - rho_e_ll)
    else
      densStar = rho_rr*sMu_R / (Ssr-SStar)
      enerStar = e_rr + (SStar - vel_R) * (SStar + p_rr / (rho_rr * sMu_R))
      UStar1 = densStar
      UStar5 = densStar*enerStar
      if orientation == 1 # x-direction
        UStar2 = densStar*SStar
        UStar3 = densStar*v2_rr
        UStar4 = densStar*v3_rr
      elseif orientation == 2 # y-direction
        UStar2 = densStar*v1_rr
        UStar3 = densStar*SStar
        UStar4 = densStar*v3_rr
      else # z-direction
        UStar2 = densStar*v1_rr
        UStar3 = densStar*v2_rr
        UStar4 = densStar*SStar
      end
      f1 = f_rr[1]+Ssr*(UStar1 - rho_rr)
      f2 = f_rr[2]+Ssr*(UStar2 - rho_v1_rr)
      f3 = f_rr[3]+Ssr*(UStar3 - rho_v2_rr)
      f4 = f_rr[4]+Ssr*(UStar4 - rho_v3_rr)
      f5 = f_rr[5]+Ssr*(UStar5 - rho_e_rr)
    end
  end
  return SVector(f1, f2, f3, f4, f5)
end



# Determine maximum stable time step based on polynomial degree and CFL number
function calc_max_dt(u, element_id, invjacobian, cfl,
                     equation::CompressibleEulerEquations3D, dg)
  λ_max = 0.0
  for k in 1:nnodes(dg), j in 1:nnodes(dg), i in 1:nnodes(dg)
    rho, rho_v1, rho_v2, rho_v3, rho_e = get_node_vars(u, dg, i, j, k, element_id)
    v1 = rho_v1 / rho
    v2 = rho_v2 / rho
    v3 = rho_v3 / rho
    v_mag = sqrt(v1^2 + v2^2 + v3^2)
    p = (equation.gamma - 1) * (rho_e - 1/2 * rho * v_mag^2)
    c = sqrt(equation.gamma * p / rho)
    λ_max = max(λ_max, v_mag + c)
  end

  dt = cfl * 2 / (nnodes(dg) * invjacobian * λ_max)

  return dt
end


# Convert conservative variables to primitive
@inline function cons2prim(u, equation::CompressibleEulerEquations3D)
  rho, rho_v1, rho_v2, rho_v3, rho_e = u

  v1 = rho_v1 / rho
  v2 = rho_v2 / rho
  v3 = rho_v3 / rho
  p = (equation.gamma - 1) * (rho_e - 0.5 * rho * (v1^2 + v2^2 + v3^2))

  return SVector(rho, v1, v2, v3, p)
end

# Convert conservative variables to entropy
@inline function cons2entropy(u, equation::CompressibleEulerEquations3D)
  rho, rho_v1, rho_v2, rho_v3, rho_e = u

  v1 = rho_v1 / rho
  v2 = rho_v2 / rho
  v3 = rho_v3 / rho
  v_square = v1^2 + v2^2 + v3^2
  p = (equation.gamma - 1) * (rho_e - 0.5 * rho * v_square)
  s = log(p) - equation.gamma*log(rho)
  rho_p = rho / p

  w1 = (equation.gamma - s) / (equation.gamma-1) - 0.5 * rho_p * v_square
  w2 = rho_p * v1
  w3 = rho_p * v2
  w4 = rho_p * v3
  w5 = -rho_p

  return SVector(w1, w2, w3, w4, w5)
end


# Convert primitive to conservative variables
@inline function prim2cons(prim, equation::CompressibleEulerEquations3D)
  rho, v1, v2, v3, p = prim
  rho_v1 = rho * v1
  rho_v2 = rho * v2
  rho_v3 = rho * v3
  rho_e  = p/(equation.gamma-1) + 0.5 * (rho_v1 * v1 + rho_v2 * v2 + rho_v3 * v3)
  return SVector(rho, rho_v1, rho_v2, rho_v3, rho_e)
end


@inline function density(u, equation::CompressibleEulerEquations3D)
  rho = u[1]
  return rho
end


@inline function pressure(u, equation::CompressibleEulerEquations3D)
  rho, rho_v1, rho_v2, rho_v3, rho_e = u
  p = (equation.gamma - 1) * (rho_e - 0.5 * (rho_v1^2 + rho_v2^2 + rho_v3^2) / rho)
  return p
end


@inline function density_pressure(u, equation::CompressibleEulerEquations3D)
 rho, rho_v1, rho_v2, rho_v3, rho_e = u
 rho_times_p = (equation.gamma - 1) * (rho * rho_e - 0.5 * (rho_v1^2 + rho_v2^2 + rho_v3^2))
 return rho_times_p
end


# Calculate thermodynamic entropy for a conservative state `cons`
@inline function entropy_thermodynamic(cons, equation::CompressibleEulerEquations3D)
  # Pressure
  p = (equation.gamma - 1) * (cons[5] - 1/2 * (cons[2]^2 + cons[3]^2 + cons[4]^2) / cons[1])

  # Thermodynamic entropy
  s = log(p) - equation.gamma*log(cons[1])

  return s
end


# Calculate mathematical entropy for a conservative state `cons`
@inline function entropy_math(cons, equation::CompressibleEulerEquations3D)
  S = -entropy_thermodynamic(cons, equation) * cons[1] / (equation.gamma - 1)
  # Mathematical entropy

  return S
end


# Default entropy is the mathematical entropy
@inline entropy(cons, equation::CompressibleEulerEquations3D) = entropy_math(cons, equation)


# Calculate total energy for a conservative state `cons`
@inline energy_total(cons, ::CompressibleEulerEquations3D) = cons[5]


# Calculate kinetic energy for a conservative state `cons`
@inline function energy_kinetic(cons, equation::CompressibleEulerEquations3D)
  return 0.5 * (cons[2]^2 + cons[3]^2 + cons[4]^2)/cons[1]
end


# Calculate internal energy for a conservative state `cons`
@inline function energy_internal(cons, equation::CompressibleEulerEquations3D)
  return energy_total(cons, equation) - energy_kinetic(cons, equation)
end
