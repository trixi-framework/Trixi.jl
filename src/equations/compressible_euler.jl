
@doc raw"""
    CompressibleEulerEquations

The compressible Euler equations for an ideal gas in two space dimensions.
"""
struct CompressibleEulerEquations <: AbstractEquation{4}
  gamma::Float64
end

function CompressibleEulerEquations()
  gamma = parameter("gamma", 1.4)

  CompressibleEulerEquations(gamma)
end


get_name(::CompressibleEulerEquations) = "CompressibleEulerEquations"
varnames_cons(::CompressibleEulerEquations) = @SVector ["rho", "rho_v1", "rho_v2", "rho_e"]
varnames_prim(::CompressibleEulerEquations) = @SVector ["rho", "v1", "v2", "p"]


# Set initial conditions at physical location `x` for time `t`
function initial_conditions_density_pulse(x, t, equation::CompressibleEulerEquations)
  rho = 1 + exp(-(x[1]^2 + x[2]^2))/2
  v1 = 1
  v2 = 1
  rho_v1 = rho * v1
  rho_v2 = rho * v2
  p = 1
  rho_e = p/(equation.gamma - 1) + 1/2 * rho * (v1^2 + v2^2)
  return @SVector [rho, rho_v1, rho_v2, rho_e]
end

function initial_conditions_pressure_pulse(x, t, equation::CompressibleEulerEquations)
  rho = 1
  v1 = 1
  v2 = 1
  rho_v1 = rho * v1
  rho_v2 = rho * v2
  p = 1 + exp(-(x[1]^2 + x[2]^2))/2
  rho_e = p/(equation.gamma - 1) + 1/2 * rho * (v1^2 + v2^2)
  return @SVector [rho, rho_v1, rho_v2, rho_e]
end

function initial_conditions_density_pressure_pulse(x, t, equation::CompressibleEulerEquations)
  rho = 1 + exp(-(x[1]^2 + x[2]^2))/2
  v1 = 1
  v2 = 1
  rho_v1 = rho * v1
  rho_v2 = rho * v2
  p = 1 + exp(-x^2)/2
  rho_e = p/(equation.gamma - 1) + 1/2 * rho * (v1^2 + v2^2)
  return @SVector [rho, rho_v1, rho_v2, rho_e]
end

function initial_conditions_constant(x, t, equation::CompressibleEulerEquations)
  rho = 1.0
  rho_v1 = 0.1
  rho_v2 = -0.2
  rho_e = 10.0
  return @SVector [rho, rho_v1, rho_v2, rho_e]
end

function initial_conditions_convergence_test(x, t, equation::CompressibleEulerEquations)
  c = 2
  A = 0.1
  L = 2
  f = 1/L
  ω = 2 * pi * f
  ini = c + A * sin(ω * (x[1] + x[2] - t))

  rho = ini
  rho_v1 = ini
  rho_v2 = ini
  rho_e = ini^2

  return @SVector [rho, rho_v1, rho_v2, rho_e]
end

function initial_conditions_sod(x, t, equation::CompressibleEulerEquations)
  if x < 0.0
    return @SVector [1.0, 0.0, 0.0, 2.5]
  else
    return @SVector [0.125, 0.0, 0.0, 0.25]
  end
end

function initial_conditions_isentropic_vortex(x, t, equation::CompressibleEulerEquations)
  # needs appropriate mesh size, e.g. [-10,-10]x[10,10]
  # make sure that the inicenter does not exit the domain, e.g. T=10.0
  # initial center of the vortex
  inicenter = [0,0]
  # size and strength of the vortex
  iniamplitude = 0.2
  # base flow
  prim=[1.0,1.0,1.0,10.0]
  vel=prim[2:3]
  rt=prim[4]/prim[1]                      # ideal gas equation
  cent=(inicenter+vel*t)                  # advection of center
  cent=x-cent                             # distance to centerpoint
  #cent=cross(iniaxis,cent)               # distance to axis, tangent vector, length r
  # cross product with iniaxis = [0,0,1]
  helper =  cent[1]
  cent[1] = -cent[2]
  cent[2] = helper
  r2=cent[1]^2+cent[2]^2
  du = iniamplitude/(2*π)*exp(0.5*(1-r2)) # vel. perturbation
  dtemp = -(equation.gamma-1)/(2*equation.gamma*rt)*du^2            # isentrop
  prim[1]=prim[1]*(1+dtemp)^(1\(equation.gamma-1))
  prim[2:3]=prim[2:3]+du*cent #v
  prim[4]=prim[4]*(1+dtemp)^(equation.gamma/(equation.gamma-1))
  rho,rho_v1,rho_v2,rho_e = prim2cons(prim, equation)
  return @SVector [rho, rho_v1, rho_v2, rho_e]
end

function initial_conditions_weak_blast_wave(x, t, equation::CompressibleEulerEquations)
  # From Hennemann & Gassner JCP paper 2020 (Sec. 6.3)
  # Set up polar coordinates
  inicenter = [0, 0]
  x_norm = x[1] - inicenter[1]
  y_norm = x[2] - inicenter[2]
  r = sqrt(x_norm^2 + y_norm^2)
  phi = atan(y_norm, x_norm)

  # Calculate primitive variables
  rho = r > 0.5 ? 1.0 : 1.1691
  v1 = r > 0.5 ? 0.0 : 0.1882 * cos(phi)
  v2 = r > 0.5 ? 0.0 : 0.1882 * sin(phi)
  p = r > 0.5 ? 1.0 : 1.245

  return prim2cons(SVector(rho, v1, v2, p), equation)
end

function initial_conditions_blast_wave(x, t, equation::CompressibleEulerEquations)
  # Modified From Hennemann & Gassner JCP paper 2020 (Sec. 6.3) -> "medium blast wave"
  # Set up polar coordinates
  inicenter = [0, 0]
  x_norm = x[1] - inicenter[1]
  y_norm = x[2] - inicenter[2]
  r = sqrt(x_norm^2 + y_norm^2)
  phi = atan(y_norm, x_norm)

  # Calculate primitive variables
  rho = r > 0.5 ? 1.0 : 1.1691
  v1 = r > 0.5 ? 0.0 : 0.1882 * cos(phi)
  v2 = r > 0.5 ? 0.0 : 0.1882 * sin(phi)
  p = r > 0.5 ? 1.0E-3 : 1.245

  return prim2cons(SVector(rho, v1, v2, p), equation)
end

function initial_conditions_sedov_blast_wave(x, t, equation::CompressibleEulerEquations)
  # Set up polar coordinates
  inicenter = [0, 0]
  x_norm = x[1] - inicenter[1]
  y_norm = x[2] - inicenter[2]
  r = sqrt(x_norm^2 + y_norm^2)

  # Setup based on http://flash.uchicago.edu/site/flashcode/user_support/flash_ug_devel/node184.html#SECTION010114000000000000000
  r0 = 0.21875 # = 3.5 * smallest dx (for domain length=4 and max-ref=6)
  # r0 = 0.5 # = more reasonable setup
  E = 1.0
  p0_inner = 3 * (equation.gamma - 1) * E / (3 * pi * r0^2)
  p0_outer = 1.0e-5 # = true Sedov setup
  # p0_outer = 1.0e-3 # = more reasonable setup

  # Calculate primitive variables
  rho = 1.0
  v1 = 0.0
  v2 = 0.0
  p = r > r0 ? p0_outer : p0_inner

  return prim2cons(SVector(rho, v1, v2, p), equation)
end

function initial_conditions_medium_sedov_blast_wave(x, t, equation::CompressibleEulerEquations)
  # Set up polar coordinates
  inicenter = [0, 0]
  x_norm = x[1] - inicenter[1]
  y_norm = x[2] - inicenter[2]
  r = sqrt(x_norm^2 + y_norm^2)

  # Setup based on http://flash.uchicago.edu/site/flashcode/user_support/flash_ug_devel/node184.html#SECTION010114000000000000000
  r0 = 0.21875 # = 3.5 * smallest dx (for domain length=4 and max-ref=6)
  # r0 = 0.5 # = more reasonable setup
  E = 1.0
  p0_inner = 3 * (equation.gamma - 1) * E / (3 * pi * r0^2)
  # p0_outer = 1.0e-5 # = true Sedov setup
  p0_outer = 1.0e-3 # = more reasonable setup

  # Calculate primitive variables
  rho = 1.0
  v1 = 0.0
  v2 = 0.0
  p = r > r0 ? p0_outer : p0_inner

  return prim2cons(SVector(rho, v1, v2, p), equation)
end

function initial_conditions_khi(x, t, equation::CompressibleEulerEquations)
  # https://rsaa.anu.edu.au/research/established-projects/fyris/2-d-kelvin-helmholtz-test
  # change discontinuity to tanh
  # typical resolution 128^2, 256^2
  # domain size is [-0.5,0.5]^2
  dens0 = 1.0 # outside density
  dens1 = 2.0 # inside density
  velx0 = -0.5 # outside velocity
  velx1 = 0.5 # inside velocity
  slope = 50 # used for tanh instead of discontinuous initial condition
  # pressure equilibrium
  p     = 2.5
  #  y velocity v2 is only white noise
  v2  = 0.01*(rand(Float64,1)[1]-0.5)
  # density
  rho = dens0 + (dens1-dens0) * 0.5*(1+(tanh(slope*(x[2]+0.25)) - (tanh(slope*(x[2]-0.25)) + 1)))
  #  x velocity is also augmented with noise
  v1 = velx0 + (velx1-velx0) * 0.5*(1+(tanh(slope*(x[2]+0.25)) - (tanh(slope*(x[2]-0.25)) + 1)))+0.01*(rand(Float64,1)[1]-0.5)
  return prim2cons(SVector(rho, v1, v2, p), equation)
end

function initial_conditions_blob(x, t, equation::CompressibleEulerEquations)
  # blob test case, see Agertz et al. https://arxiv.org/pdf/astro-ph/0610051.pdf
  # other reference: https://arxiv.org/pdf/astro-ph/0610051.pdf
  # change discontinuity to tanh
  # typical domain is rectangular, we change it to a square, as Trixi can only do squares
  # resolution 128^2, 256^2
  # domain size is [-20.0,20.0]^2
  # gamma = 5/3 for this test case
  R = 1.0 # radius of the blob
  # background density
  dens0 = 1.0
  Chi = 10.0 # density contrast
  # reference time of characteristic growth of KH instability equal to 1.0
  tau_kh = 1.0
  tau_cr = tau_kh/1.6 # crushing time
  # determine background velocity
  velx0 = 2*R*sqrt(Chi)/tau_cr
  vely0 = 0.0
  Ma0 = 2.7 # background flow Mach number Ma=v/c
  c = velx0/Ma0 # sound speed
  # use perfect gas assumption to compute background pressure via the sound speed c^2 = gamma * pressure/density
  p0 = c*c*dens0/equation.gamma
  # initial center of the blob
  inicenter = [-15,0]
  x_rel = x-inicenter
  r = sqrt(x_rel[1]^2 + x_rel[2]^2)
  # steepness of the tanh transition zone
  slope = 2
  # density blob
  dens = dens0 + (Chi-1) * 0.5*(1+(tanh(slope*(r+R)) - (tanh(slope*(r-R)) + 1)))
  # velocity blob is zero
  velx = velx0 - velx0 * 0.5*(1+(tanh(slope*(r+R)) - (tanh(slope*(r-R)) + 1)))
  return prim2cons(SVector(dens, velx, vely0, p0), equation)
end


# Apply source terms
function source_terms_convergence_test(ut, u, x, element_id, t, n_nodes, equation::CompressibleEulerEquations)
  # Same settings as in `initial_conditions`
  c = 2
  A = 0.1
  L = 2
  f = 1/L
  ω = 2 * pi * f
  γ = equation.gamma

  for j in 1:n_nodes
    for i in 1:n_nodes
      x1 = x[1, i, j, element_id]
      x2 = x[2, i, j, element_id]
      tmp1 = cos((x1 + x2 - t)*ω)*A*ω
      tmp2 = sin((x1 + x2 - t)*ω)*A
      tmp3 = γ - 1
      tmp4 = (2*c - 1)*tmp3
      tmp5 = (2*tmp2*γ - 2*tmp2 + tmp4 + 1)*tmp1
      tmp6 = tmp2 + c

      ut[1, i, j, element_id] += tmp1
      ut[2, i, j, element_id] += tmp5
      ut[3, i, j, element_id] += tmp5
      ut[4, i, j, element_id] += 2*((tmp6 - 1)*tmp3 + tmp6*γ)*tmp1

      # Original terms (without performanc enhancements)
      # ut[1, i, j, element_id] += cos((x1 + x2 - t)*ω)*A*ω
      # ut[2, i, j, element_id] += (2*sin((x1 + x2 - t)*ω)*A*γ - 2*sin((x1 + x2 - t)*ω)*A +
      #                             2*c*γ - 2*c - γ + 2)*cos((x1 + x2 - t)*ω)*A*ω
      # ut[3, i, j, element_id] += (2*sin((x1 + x2 - t)*ω)*A*γ - 2*sin((x1 + x2 - t)*ω)*A +
      #                             2*c*γ - 2*c - γ + 2)*cos((x1 + x2 - t)*ω)*A*ω
      # ut[4, i, j, element_id] += 2*((c - 1 + sin((x1 + x2 - t)*ω)*A)*(γ - 1) +
      #                               (sin((x1 + x2 - t)*ω)*A + c)*γ)*cos((x1 + x2 - t)*ω)*A*ω
    end
  end

  return nothing
end


# Calculate 1D flux in for a single point
@inline function calcflux(u, orientation, equation::CompressibleEulerEquations)
  rho, rho_v1, rho_v2, rho_e = u
  v1 = rho_v1/rho
  v2 = rho_v2/rho
  p = (equation.gamma - 1) * (rho_e - 1/2 * rho * (v1^2 + v2^2))
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


"""
    function flux_kuya_etal(u_ll, u_rr, orientation, equation::CompressibleEulerEquations)

Kinetic energy preserving two-point flux with pressure oscillation fix
by Kuya, Totani and Kawai (2018)
  Kinetic energy and entropy preserving schemes for compressible flows
  by split convective forms
[DOI: 10.1016/j.jcp.2018.08.058](https://doi.org/10.1016/j.jcp.2018.08.058)
"""
@inline function flux_kuya_etal(u_ll, u_rr, orientation, equation::CompressibleEulerEquations)
  # Unpack left and right state
  rho_ll, rho_v1_ll, rho_v2_ll, rho_e_ll = u_ll
  rho_rr, rho_v1_rr, rho_v2_rr, rho_e_rr = u_rr

  v1_ll = rho_v1_ll/rho_ll
  v2_ll = rho_v2_ll/rho_ll
  v1_rr = rho_v1_rr/rho_rr
  v2_rr = rho_v2_rr/rho_rr
  p_ll =  (equation.gamma - 1) * (rho_e_ll - 1/2 * rho_ll * (v1_ll^2 + v2_ll^2))
  p_rr =  (equation.gamma - 1) * (rho_e_rr - 1/2 * rho_rr * (v1_rr^2 + v2_rr^2))

  # Average each factor of products in flux
  rho_avg = 1/2 * (rho_ll + rho_rr)
  v1_avg = 1/2 * (v1_ll + v1_rr)
  v2_avg = 1/2 * (v2_ll + v2_rr)
  p_avg = 1/2 * (p_ll + p_rr)
  kin_avg = 1/2 * (v1_ll*v1_rr + v2_ll*v2_rr)

  # Calculate fluxes depending on orientation
  if orientation == 1
    pv1_avg = 1/2 * ( p_ll*v1_ll + p_rr*v1_rr)
    f1 = rho_avg * v1_avg
    f2 = rho_avg * v1_avg * v1_avg + p_avg
    f3 = rho_avg * v1_avg * v2_avg
    f4 = p_avg*v1_avg/(equation.gamma-1) + rho_avg*v1_avg*kin_avg + pv1_avg
  else
    pv2_avg = 1/2 * ( p_ll*v2_ll + p_rr*v2_rr)
    f1 = rho_avg * v2_avg
    f2 = rho_avg * v2_avg * v1_avg
    f3 = rho_avg * v2_avg * v2_avg + p_avg
    f4 = p_avg*v2_avg/(equation.gamma-1) + rho_avg*v2_avg*kin_avg + pv2_avg
  end

  return SVector(f1, f2, f3, f4)
end


"""
    flux_kennedy_gruber(u_ll, u_rr, orientation, equation::CompressibleEulerEquations)

Kinetic energy preserving two-point flux by Kennedy and Gruber (2008)
  Reduced aliasing formulations of the convective terms within the
  Navier-Stokes equations for a compressible fluid
[DOI: 10.1016/j.jcp.2007.09.020](https://doi.org/10.1016/j.jcp.2007.09.020)
"""
@inline function flux_kennedy_gruber(u_ll, u_rr, orientation, equation::CompressibleEulerEquations)
  # Unpack left and right state
  rho_ll, rho_v1_ll, rho_v2_ll, rho_e_ll = u_ll
  rho_rr, rho_v1_rr, rho_v2_rr, rho_e_rr = u_rr

  v1_ll = rho_v1_ll/rho_ll
  v2_ll = rho_v2_ll/rho_ll
  v1_rr = rho_v1_rr/rho_rr
  v2_rr = rho_v2_rr/rho_rr

  # Average each factor of products in flux
  rho_avg = 1/2 * (rho_ll + rho_rr)
  v1_avg = 1/2 * (v1_ll + v1_rr)
  v2_avg = 1/2 * (v2_ll + v2_rr)
  p_avg = 1/2 * ((equation.gamma - 1) * (rho_e_ll - 1/2 * rho_ll * (v1_ll^2 + v2_ll^2)) +
                 (equation.gamma - 1) * (rho_e_rr - 1/2 * rho_rr * (v1_rr^2 + v2_rr^2)))
  e_avg = 1/2 * (rho_e_ll/rho_ll + rho_e_rr/rho_rr)

  # Calculate fluxes depending on orientation
  if orientation == 1
    f1 = rho_avg * v1_avg
    f2 = rho_avg * v1_avg * v1_avg + p_avg
    f3 = rho_avg * v1_avg * v2_avg
    f4 = (rho_avg * e_avg + p_avg) * v1_avg
  else
    f1 = rho_avg * v2_avg
    f2 = rho_avg * v2_avg * v1_avg
    f3 = rho_avg * v2_avg * v2_avg + p_avg
    f4 = (rho_avg * e_avg + p_avg) * v2_avg
  end

  return SVector(f1, f2, f3, f4)
end


"""
    flux_chandrashekar(u_ll, u_rr, orientation, equation::CompressibleEulerEquations)

Entropy conserving two-point flux by Chandrashekar (2013)
  Kinetic Energy Preserving and Entropy Stable Finite Volume Schemes
  for Compressible Euler and Navier-Stokes Equations
[DOI: 10.4208/cicp.170712.010313a](https://doi.org/10.4208/cicp.170712.010313a)
"""
@inline function flux_chandrashekar(u_ll, u_rr, orientation, equation::CompressibleEulerEquations)
  # Unpack left and right state
  rho_ll, rho_v1_ll, rho_v2_ll, rho_e_ll = u_ll
  rho_rr, rho_v1_rr, rho_v2_rr, rho_e_rr = u_rr

  v1_ll = rho_v1_ll/rho_ll
  v2_ll = rho_v2_ll/rho_ll
  v1_rr = rho_v1_rr/rho_rr
  v2_rr = rho_v2_rr/rho_rr
  p_ll =  (equation.gamma - 1) * (rho_e_ll - 1/2 * rho_ll * (v1_ll^2 + v2_ll^2))
  p_rr =  (equation.gamma - 1) * (rho_e_rr - 1/2 * rho_rr * (v1_rr^2 + v2_rr^2))
  beta_ll = 0.5*rho_ll/p_ll
  beta_rr = 0.5*rho_rr/p_rr
  specific_kin_ll = 0.5*(v1_ll^2 + v2_ll^2)
  specific_kin_rr = 0.5*(v1_rr^2 + v2_rr^2)

  # Compute the necessary mean values
  rho_avg  = 0.5*(rho_ll+rho_rr)
  rho_mean = ln_mean(rho_ll,rho_rr)
  beta_mean = ln_mean(beta_ll,beta_rr)
  beta_avg = 0.5*(beta_ll+beta_rr)
  v1_avg = 0.5*(v1_ll+v1_rr)
  v2_avg = 0.5*(v2_ll+v2_rr)
  p_mean = 0.5*rho_avg/beta_avg
  velocity_square_avg = specific_kin_ll + specific_kin_rr

  # Calculate fluxes depending on orientation
  if orientation == 1
    f1 = rho_mean * v1_avg
    f2 = f1 * v1_avg + p_mean
    f3 = f1 * v2_avg
    f4 = f1 * 0.5*(1/(equation.gamma-1)/beta_mean - velocity_square_avg)+f2*v1_avg + f3*v2_avg
  else
    f1 = rho_mean * v2_avg
    f2 = f1 * v1_avg
    f3 = f1 * v2_avg + p_mean
    f4 = f1 * 0.5*(1/(equation.gamma-1)/beta_mean - velocity_square_avg)+f2*v1_avg + f3*v2_avg
  end

  return SVector(f1, f2, f3, f4)
end


function flux_lax_friedrichs(u_ll, u_rr, orientation, equation::CompressibleEulerEquations)
  # Calculate primitive variables and speed of sound
  rho_ll, rho_v1_ll, rho_v2_ll, rho_e_ll = u_ll
  rho_rr, rho_v1_rr, rho_v2_rr, rho_e_rr = u_rr

  v1_ll = rho_v1_ll / rho_ll
  v2_ll = rho_v2_ll / rho_ll
  v_mag_ll = sqrt(v1_ll^2 + v2_ll^2)
  p_ll = (equation.gamma - 1) * (rho_e_ll - 1/2 * rho_ll * v_mag_ll^2)
  c_ll = sqrt(equation.gamma * p_ll / rho_ll)
  v1_rr = rho_v1_rr / rho_rr
  v2_rr = rho_v2_rr / rho_rr
  v_mag_rr = sqrt(v1_rr^2 + v2_rr^2)
  p_rr = (equation.gamma - 1) * (rho_e_rr - 1/2 * rho_rr * v_mag_rr^2)
  c_rr = sqrt(equation.gamma * p_rr / rho_rr)

  # Obtain left and right fluxes
  f_ll = calcflux(u_ll, orientation, equation)
  f_rr = calcflux(u_rr, orientation, equation)

  λ_max = max(v_mag_ll, v_mag_rr) + max(c_ll, c_rr)
  f1 = 1/2 * (f_ll[1] + f_rr[1]) - 1/2 * λ_max * (rho_rr    - rho_ll)
  f2 = 1/2 * (f_ll[2] + f_rr[2]) - 1/2 * λ_max * (rho_v1_rr - rho_v1_ll)
  f3 = 1/2 * (f_ll[3] + f_rr[3]) - 1/2 * λ_max * (rho_v2_rr - rho_v2_ll)
  f4 = 1/2 * (f_ll[4] + f_rr[4]) - 1/2 * λ_max * (rho_e_rr  - rho_e_ll)

  return SVector(f1, f2, f3, f4)
end


# Determine maximum stable time step based on polynomial degree and CFL number
function calc_max_dt(u, element_id, n_nodes, invjacobian, cfl,
                     equation::CompressibleEulerEquations)
  λ_max = 0.0
  for j = 1:n_nodes
    for i = 1:n_nodes
      rho    = u[1, i, j, element_id]
      rho_v1 = u[2, i, j, element_id]
      rho_v2 = u[3, i, j, element_id]
      rho_e  = u[4, i, j, element_id]
      v1 = rho_v1/rho
      v2 = rho_v2/rho
      v_mag = sqrt(v1^2 + v2^2)
      p = (equation.gamma - 1) * (rho_e - 1/2 * rho * v_mag^2)
      c = sqrt(equation.gamma * p / rho)
      λ_max = max(λ_max, v_mag + c)
    end
  end

  dt = cfl * 2 / (invjacobian * λ_max) / n_nodes

  return dt
end


# Convert conservative variables to primitive
function cons2prim(cons, equation::CompressibleEulerEquations)
  prim = similar(cons)
  @. prim[1, :, :, :] = cons[1, :, :, :]
  @. prim[2, :, :, :] = cons[2, :, :, :] / cons[1, :, :, :]
  @. prim[3, :, :, :] = cons[3, :, :, :] / cons[1, :, :, :]
  @. prim[4, :, :, :] = ((equation.gamma - 1)
                         * (cons[4, :, :, :] - 1/2 * (cons[2, :, :, :] * prim[2, :, :, :] +
                                                      cons[3, :, :, :] * prim[3, :, :, :])))
  return prim
end

# Convert conservative variables to entropy
function cons2entropy(cons, n_nodes, n_elements, equation::CompressibleEulerEquations)
  entropy = similar(cons)
  v = zeros(2,n_nodes,n_nodes,n_elements)
  v_square = zeros(n_nodes,n_nodes,n_elements)
  p = zeros(n_nodes,n_nodes,n_elements)
  s = zeros(n_nodes,n_nodes,n_elements)
  rho_p = zeros(n_nodes,n_nodes,n_elements)

  @. v[1, :, :, :] = cons[2, :, :, :] / cons[1, :, :, :]
  @. v[2, :, :, :] = cons[3, :, :, :] / cons[1, :, :, :]
  @. v_square[ :, :, :] = v[1, :, :, :]*v[1, :, :, :]+v[2, :, :, :]*v[2, :, :, :]
  @. p[ :, :, :] = ((equation.gamma - 1)
                         * (cons[4, :, :, :] - 1/2 * (cons[2, :, :, :] * v[1, :, :, :] +
                            cons[3, :, :, :] * v[2, :, :, :])))
  @. s[ :, :, :] = log(p[:, :, :]) - equation.gamma*log(cons[1, :, :, :])
  @. rho_p[ :, :, :] = cons[1, :, :, :] / p[ :, :, :]

  @. entropy[1, :, :, :] = (equation.gamma - s[:,:,:])/(equation.gamma-1) -
                           0.5*rho_p[:,:,:]*v_square[:,:,:]
  @. entropy[2, :, :, :] = rho_p[:,:,:]*v[1,:,:,:]
  @. entropy[3, :, :, :] = rho_p[:,:,:]*v[2,:,:,:]
  @. entropy[4, :, :, :] = -rho_p[:,:,:]

  return entropy
end


# Convert primitive to conservative variables
function prim2cons(prim, equation::CompressibleEulerEquations)
  cons = similar(prim)
  cons[1] = prim[1]
  cons[2] = prim[2] * prim[1]
  cons[3] = prim[3] * prim[1]
  cons[4] = prim[4]/(equation.gamma-1)+1/2*(cons[2] * prim[2] + cons[3] * prim[3])
  return cons
end


# Convert conservative variables to indicator variable for discontinuities (elementwise version)
@inline function cons2indicator!(indicator, cons, element_id, n_nodes, indicator_variable,
                                 equation::CompressibleEulerEquations)
  for j in 1:n_nodes
    for i in 1:n_nodes
      indicator[1, i, j] = cons2indicator(cons[1, i, j, element_id], cons[2, i, j, element_id],
                                          cons[3, i, j, element_id], cons[4, i, j, element_id],
                                          indicator_variable, equation)
    end
  end
end


# Convert conservative variables to indicator variable for discontinuities (pointwise version)
@inline function cons2indicator(rho, rho_v1, rho_v2, rho_e, ::Val{:density},
                                equation::CompressibleEulerEquations)
  # Indicator variable is rho
  return rho
end


# Convert conservative variables to indicator variable for discontinuities (pointwise version)
@inline function cons2indicator(rho, rho_v1, rho_v2, rho_e, ::Val{:density_pressure},
                                equation::CompressibleEulerEquations)
  v1 = rho_v1/rho
  v2 = rho_v2/rho

  # Calculate pressure
  p = (equation.gamma - 1) * (rho_e - 1/2 * rho * (v1^2 + v2^2))

  # Indicator variable is rho * p
  return rho * p
end


# Convert conservative variables to indicator variable for discontinuities (pointwise version)
@inline function cons2indicator(rho, rho_v1, rho_v2, rho_e, ::Val{:pressure},
                                equation::CompressibleEulerEquations)
  v1 = rho_v1/rho
  v2 = rho_v2/rho

  # Indicator variable is p
  return (equation.gamma - 1) * (rho_e - 1/2 * rho * (v1^2 + v2^2))
end


# Calculates the entropy flux in direction "orientation" and the entropy variables for a state cons
@inline function cons2entropyvars_and_flux(gamma::Float64, cons, orientation::Int)
  entropy = MVector{4, Float64}(undef)
  v = (cons[2] / cons[1] , cons[3] / cons[1])
  v_square= v[1]*v[1]+v[2]*v[2]
  p = (gamma - 1) * (cons[4] - 1/2 * (cons[2] * v[1] + cons[3] * v[2]))
  rho_p = cons[1] / p
  # thermodynamic entropy
  s = log(p) - gamma*log(cons[1])
  # mathematical entropy
  S = - s*cons[1]/(gamma-1)
  # entropy variables
  entropy[1] = (gamma - s)/(gamma-1) - 0.5*rho_p*v_square
  entropy[2] = rho_p*v[1]
  entropy[3] = rho_p*v[2]
  entropy[4] = -rho_p
  # entropy flux
  entropy_flux = S*v[orientation]
  return entropy, entropy_flux
end


# Calculate thermodynamic entropy for a conservative state `cons`
@inline function entropy_thermodynamic(cons, equation::CompressibleEulerEquations)
  # Pressure
  p = (equation.gamma - 1) * (cons[4] - 1/2 * (cons[2]^2 + cons[3]^2) / cons[1])

  # Thermodynamic entropy
  s = log(p) - equation.gamma*log(cons[1])

  return s
end


# Calculate mathematical entropy for a conservative state `cons`
@inline function entropy_math(cons, equation::CompressibleEulerEquations)
  # Mathematical entropy
  S = -entropy_thermodynamic(cons, equation) * cons[1] / (equation.gamma - 1)

  return S
end


# Default entropy is the mathematical entropy
@inline entropy(cons, equation::CompressibleEulerEquations) = entropy_math(cons, equation)


# Calculate total energy for a conservative state `cons`
@inline energy_total(cons, ::CompressibleEulerEquations) = cons[4]


# Calculate kinetic energy for a conservative state `cons`
@inline function energy_kinetic(cons, equation::CompressibleEulerEquations)
  return 0.5 * (cons[2]^2 + cons[3]^2)/cons[1]
end


# Calculate internal energy for a conservative state `cons`
@inline function energy_internal(cons, equation::CompressibleEulerEquations)
  return energy_total(cons, equation) - energy_kinetic(cons, equation)
end
