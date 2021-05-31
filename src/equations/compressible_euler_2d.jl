
@doc raw"""
    CompressibleEulerEquations2D

The compressible Euler equations for an ideal gas in two space dimensions.
"""
struct CompressibleEulerEquations2D{RealT<:Real} <: AbstractCompressibleEulerEquations{2, 4}
  gamma::RealT
end


varnames(::typeof(cons2cons), ::CompressibleEulerEquations2D) = ("rho", "rho_v1", "rho_v2", "rho_e")
varnames(::typeof(cons2prim), ::CompressibleEulerEquations2D) = ("rho", "v1", "v2", "p")


# Set initial conditions at physical location `x` for time `t`
"""
    initial_condition_constant(x, t, equations::CompressibleEulerEquations2D)

A constant initial condition to test free-stream preservation.
"""
function initial_condition_constant(x, t, equations::CompressibleEulerEquations2D)
  rho = 1.0
  rho_v1 = 0.1
  rho_v2 = -0.2
  rho_e = 10.0
  return SVector(rho, rho_v1, rho_v2, rho_e)
end


"""
    initial_condition_convergence_test(x, t, equations::CompressibleEulerEquations2D)

A smooth initial condition used for convergence tests in combination with
[`source_terms_convergence_test`](@ref)
(and [`BoundaryConditionDirichlet(initial_condition_convergence_test)`](@ref) in non-periodic domains).
"""
function initial_condition_convergence_test(x, t, equations::CompressibleEulerEquations2D)
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

  return SVector(rho, rho_v1, rho_v2, rho_e)
end

"""
    source_terms_convergence_test(u, x, t, equations::CompressibleEulerEquations2D)

Source terms used for convergence tests in combination with
[`initial_condition_convergence_test`](@ref)
(and [`BoundaryConditionDirichlet(initial_condition_convergence_test)`](@ref) in non-periodic domains).
"""
@inline function source_terms_convergence_test(u, x, t, equations::CompressibleEulerEquations2D)
  # Same settings as in `initial_condition`
  c = 2
  A = 0.1
  L = 2
  f = 1/L
  ω = 2 * pi * f
  γ = equations.gamma

  x1, x2 = x
  si, co = sincos((x1 + x2 - t)*ω)
  tmp1 = co * A * ω
  tmp2 = si * A
  tmp3 = γ - 1
  tmp4 = (2*c - 1)*tmp3
  tmp5 = (2*tmp2*γ - 2*tmp2 + tmp4 + 1)*tmp1
  tmp6 = tmp2 + c

  du1 = tmp1
  du2 = tmp5
  du3 = tmp5
  du4 = 2*((tmp6 - 1)*tmp3 + tmp6*γ)*tmp1

  # Original terms (without performanc enhancements)
  # du1 = cos((x1 + x2 - t)*ω)*A*ω
  # du2 = (2*sin((x1 + x2 - t)*ω)*A*γ - 2*sin((x1 + x2 - t)*ω)*A +
  #                             2*c*γ - 2*c - γ + 2)*cos((x1 + x2 - t)*ω)*A*ω
  # du3 = (2*sin((x1 + x2 - t)*ω)*A*γ - 2*sin((x1 + x2 - t)*ω)*A +
  #                             2*c*γ - 2*c - γ + 2)*cos((x1 + x2 - t)*ω)*A*ω
  # du3 = 2*((c - 1 + sin((x1 + x2 - t)*ω)*A)*(γ - 1) +
  #                             (sin((x1 + x2 - t)*ω)*A + c)*γ)*cos((x1 + x2 - t)*ω)*A*ω

  return SVector(du1, du2, du3, du4)
end


"""
    initial_condition_density_pulse(x, t, equations::CompressibleEulerEquations2D)

A Gaussian pulse in the density with constant velocity and pressure; reduces the
compressible Euler equations to the linear advection equations.
"""
function initial_condition_density_pulse(x, t, equations::CompressibleEulerEquations2D)
  rho = 1 + exp(-(x[1]^2 + x[2]^2))/2
  v1 = 1
  v2 = 1
  rho_v1 = rho * v1
  rho_v2 = rho * v2
  p = 1
  rho_e = p/(equations.gamma - 1) + 1/2 * rho * (v1^2 + v2^2)
  return SVector(rho, rho_v1, rho_v2, rho_e)
end


"""
    initial_condition_density_wave(x, t, equations::CompressibleEulerEquations2D)

A sine wave in the density with constant velocity and pressure; reduces the
compressible Euler equations to the linear advection equations.
This setup is the test case for stability of EC fluxes from paper
- Gregor J. Gassner, Magnus Svärd, Florian J. Hindenlang (2020)
  Stability issues of entropy-stable and/or split-form high-order schemes
  [arXiv: 2007.09026](https://arxiv.org/abs/2007.09026)
with the following parameters
- domain [-1, 1]
- mesh = 4x4
- polydeg = 5
"""
function initial_condition_density_wave(x, t, equations::CompressibleEulerEquations2D)
  v1 = 0.1
  v2 = 0.2
  rho = 1 + 0.98 * sinpi(2 * (x[1] + x[2] - t * (v1 + v2)))
  rho_v1 = rho * v1
  rho_v2 = rho * v2
  p = 20
  rho_e = p / (equations.gamma - 1) + 1/2 * rho * (v1^2 + v2^2)
  return SVector(rho, rho_v1, rho_v2, rho_e)
end


"""
    initial_condition_pressure_pulse(x, t, equations::CompressibleEulerEquations2D)

A Gaussian pulse in the pressure with constant velocity and density.
"""
function initial_condition_pressure_pulse(x, t, equations::CompressibleEulerEquations2D)
  rho = 1
  v1 = 1
  v2 = 1
  rho_v1 = rho * v1
  rho_v2 = rho * v2
  p = 1 + exp(-(x[1]^2 + x[2]^2))/2
  rho_e = p/(equations.gamma - 1) + 1/2 * rho * (v1^2 + v2^2)
  return SVector(rho, rho_v1, rho_v2, rho_e)
end


"""
    initial_condition_density_pressure_pulse(x, t, equations::CompressibleEulerEquations2D)

A Gaussian pulse in density and pressure with constant velocity.
"""
function initial_condition_density_pressure_pulse(x, t, equations::CompressibleEulerEquations2D)
  rho = 1 + exp(-(x[1]^2 + x[2]^2))/2
  v1 = 1
  v2 = 1
  rho_v1 = rho * v1
  rho_v2 = rho * v2
  p = 1 + exp(-(x[1]^2 + x[2]^2))/2
  rho_e = p/(equations.gamma - 1) + 1/2 * rho * (v1^2 + v2^2)
  return SVector(rho, rho_v1, rho_v2, rho_e)
end


"""
    initial_condition_isentropic_vortex(x, t, equations::CompressibleEulerEquations2D)

The classical isentropic vortex test case of
- Chi-Wang Shu (1997)
  Essentially Non-Oscillatory and Weighted Essentially Non-Oscillatory
  Schemes for Hyperbolic Conservation Laws
  [NASA/CR-97-206253](https://ntrs.nasa.gov/citations/19980007543)
"""
function initial_condition_isentropic_vortex(x, t, equations::CompressibleEulerEquations2D)
  # needs appropriate mesh size, e.g. [-10,-10]x[10,10]
  # make sure that the inicenter does not exit the domain, e.g. T=10.0
  # initial center of the vortex
  inicenter = SVector(0.0, 0.0)
  # size and strength of the vortex
  iniamplitude = 0.2
  # base flow
  rho = 1.0
  v1 = 1.0
  v2 = 1.0
  vel = SVector(v1, v2)
  p = 10.0
  vec = SVector(v1, v2)
  rt = p / rho                  # ideal gas equation
  cent = inicenter + vel*t      # advection of center
  cent = x - cent               # distance to centerpoint
  #cent=cross(iniaxis,cent)     # distance to axis, tangent vector, length r
  # cross product with iniaxis = [0,0,1]
  cent = SVector(-cent[2], cent[1])
  r2 = cent[1]^2 + cent[2]^2
  du = iniamplitude/(2*π)*exp(0.5*(1-r2)) # vel. perturbation
  dtemp = -(equations.gamma-1)/(2*equations.gamma*rt)*du^2            # isentrop
  rho = rho * (1+dtemp)^(1\(equations.gamma-1))
  vel = vel + du*cent
  v1, v2 = vel
  p = p * (1+dtemp)^(equations.gamma/(equations.gamma-1))
  prim = SVector(rho, v1, v2, p)
  return prim2cons(prim, equations)
end


"""
    initial_condition_weak_blast_wave(x, t, equations::CompressibleEulerEquations2D)

A weak blast wave taken from
- Sebastian Hennemann, Gregor J. Gassner (2020)
  A provably entropy stable subcell shock capturing approach for high order split form DG
  [arXiv: 2008.12044](https://arxiv.org/abs/2008.12044)
"""
function initial_condition_weak_blast_wave(x, t, equations::CompressibleEulerEquations2D)
  # From Hennemann & Gassner JCP paper 2020 (Sec. 6.3)
  # Set up polar coordinates
  inicenter = SVector(0.0, 0.0)
  x_norm = x[1] - inicenter[1]
  y_norm = x[2] - inicenter[2]
  r = sqrt(x_norm^2 + y_norm^2)
  phi = atan(y_norm, x_norm)
  sin_phi, cos_phi = sincos(phi)

  # Calculate primitive variables
  rho = r > 0.5 ? 1.0 : 1.1691
  v1  = r > 0.5 ? 0.0 : 0.1882 * cos_phi
  v2  = r > 0.5 ? 0.0 : 0.1882 * sin_phi
  p   = r > 0.5 ? 1.0 : 1.245

  return prim2cons(SVector(rho, v1, v2, p), equations)
end


"""
    initial_condition_blast_wave(x, t, equations::CompressibleEulerEquations2D)

A medium blast wave taken from
- Sebastian Hennemann, Gregor J. Gassner (2020)
  A provably entropy stable subcell shock capturing approach for high order split form DG
  [arXiv: 2008.12044](https://arxiv.org/abs/2008.12044)
"""
function initial_condition_blast_wave(x, t, equations::CompressibleEulerEquations2D)
  # Modified From Hennemann & Gassner JCP paper 2020 (Sec. 6.3) -> "medium blast wave"
  # Set up polar coordinates
  inicenter = SVector(0.0, 0.0)
  x_norm = x[1] - inicenter[1]
  y_norm = x[2] - inicenter[2]
  r = sqrt(x_norm^2 + y_norm^2)
  phi = atan(y_norm, x_norm)
  sin_phi, cos_phi = sincos(phi)

  # Calculate primitive variables
  rho = r > 0.5 ? 1.0 : 1.1691
  v1  = r > 0.5 ? 0.0 : 0.1882 * cos_phi
  v2  = r > 0.5 ? 0.0 : 0.1882 * sin_phi
  p   = r > 0.5 ? 1.0E-3 : 1.245

  return prim2cons(SVector(rho, v1, v2, p), equations)
end


"""
    initial_condition_sedov_blast_wave(x, t, equations::CompressibleEulerEquations2D)

The Sedov blast wave setup based on Flash
- http://flash.uchicago.edu/site/flashcode/user_support/flash_ug_devel/node184.html#SECTION010114000000000000000
"""
function initial_condition_sedov_blast_wave(x, t, equations::CompressibleEulerEquations2D)
  # Set up polar coordinates
  inicenter = SVector(0.0, 0.0)
  x_norm = x[1] - inicenter[1]
  y_norm = x[2] - inicenter[2]
  r = sqrt(x_norm^2 + y_norm^2)

  # Setup based on http://flash.uchicago.edu/site/flashcode/user_support/flash_ug_devel/node184.html#SECTION010114000000000000000
  r0 = 0.21875 # = 3.5 * smallest dx (for domain length=4 and max-ref=6)
  # r0 = 0.5 # = more reasonable setup
  E = 1.0
  p0_inner = 3 * (equations.gamma - 1) * E / (3 * pi * r0^2)
  p0_outer = 1.0e-5 # = true Sedov setup
  # p0_outer = 1.0e-3 # = more reasonable setup

  # Calculate primitive variables
  rho = 1.0
  v1  = 0.0
  v2  = 0.0
  p   = r > r0 ? p0_outer : p0_inner

  return prim2cons(SVector(rho, v1, v2, p), equations)
end


"""
    initial_condition_medium_sedov_blast_wave(x, t, equations::CompressibleEulerEquations2D)

The Sedov blast wave setup based on Flash
- http://flash.uchicago.edu/site/flashcode/user_support/flash_ug_devel/node184.html#SECTION010114000000000000000
with smaller strength of the initial discontinuity.
"""
function initial_condition_medium_sedov_blast_wave(x, t, equations::CompressibleEulerEquations2D)
  # Set up polar coordinates
  inicenter = SVector(0.0, 0.0)
  x_norm = x[1] - inicenter[1]
  y_norm = x[2] - inicenter[2]
  r = sqrt(x_norm^2 + y_norm^2)

  # Setup based on http://flash.uchicago.edu/site/flashcode/user_support/flash_ug_devel/node184.html#SECTION010114000000000000000
  r0 = 0.21875 # = 3.5 * smallest dx (for domain length=4 and max-ref=6)
  # r0 = 0.5 # = more reasonable setup
  E = 1.0
  p0_inner = 3 * (equations.gamma - 1) * E / (3 * pi * r0^2)
  # p0_outer = 1.0e-5 # = true Sedov setup
  p0_outer = 1.0e-3 # = more reasonable setup

  # Calculate primitive variables
  rho = 1.0
  v1  = 0.0
  v2  = 0.0
  p   = r > r0 ? p0_outer : p0_inner

  return prim2cons(SVector(rho, v1, v2, p), equations)
end


"""
    initial_condition_khi(x, t, equations::CompressibleEulerEquations2D)

The classical Kelvin-Helmholtz instability based on
- https://rsaa.anu.edu.au/research/established-projects/fyris/2-d-kelvin-helmholtz-test
"""
function initial_condition_khi(x, t, equations::CompressibleEulerEquations2D)
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
  # density
  rho = dens0 + (dens1-dens0) * 0.5*(1+(tanh(slope*(x[2]+0.25)) - (tanh(slope*(x[2]-0.25)) + 1)))
  if iszero(t) # initial condition
    #  y velocity v2 is only white noise
    v2  = 0.01*(rand(Float64,1)[1]-0.5)
    #  x velocity is also augmented with noise
    v1 = velx0 + (velx1-velx0) * 0.5*(1+(tanh(slope*(x[2]+0.25)) - (tanh(slope*(x[2]-0.25)) + 1)))+0.01*(rand(Float64,1)[1]-0.5)
  else # background values to compute reference values for CI
    v2 = 0.0
    v1 = velx0 + (velx1-velx0) * 0.5*(1+(tanh(slope*(x[2]+0.25)) - (tanh(slope*(x[2]-0.25)) + 1)))
  end
  return prim2cons(SVector(rho, v1, v2, p), equations)
end


"""
    initial_condition_blob(x, t, equations::CompressibleEulerEquations2D)

The blob test case taken from
- Agertz et al. (2006)
  Fundamental differences between SPH and grid methods
  [arXiv: astro-ph/0610051](https://arxiv.org/abs/astro-ph/0610051)
"""
function initial_condition_blob(x, t, equations::CompressibleEulerEquations2D)
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
  p0 = c*c*dens0/equations.gamma
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
  return prim2cons(SVector(dens, velx, vely0, p0), equations)
end


"""
    initial_condition_eoc_test_coupled_euler_gravity(x, t, equations::CompressibleEulerEquations2D)

Setup used for convergence tests of the Euler equations with self-gravity used in
- Michael Schlottke-Lakemper, Andrew R. Winters, Hendrik Ranocha, Gregor J. Gassner (2020)
  A purely hyperbolic discontinuous Galerkin approach for self-gravitating gas dynamics
  [arXiv: 2008.10593](https://arxiv.org/abs/2008.10593)
in combination with [`source_terms_eoc_test_coupled_euler_gravity`](@ref)
or [`source_terms_eoc_test_euler`](@ref).
"""
function initial_condition_eoc_test_coupled_euler_gravity(x, t, equations::CompressibleEulerEquations2D)
  # OBS! this assumes that γ = 2 other manufactured source terms are incorrect
  if equations.gamma != 2.0
    error("adiabatic constant must be 2 for the coupling convergence test")
  end
  c = 2.0
  A = 0.1
  ini = c + A * sin(pi * (x[1] + x[2] - t))
  G = 1.0 # gravitational constant

  rho = ini
  v1 = 1.0
  v2 = 1.0
  p = ini^2 * G / pi # * 2 / ndims, but ndims==2 here

  return prim2cons(SVector(rho, v1, v2, p), equations)
end

"""
    source_terms_eoc_test_coupled_euler_gravity(u, x, t, equations::CompressibleEulerEquations2D)

Setup used for convergence tests of the Euler equations with self-gravity used in
- Michael Schlottke-Lakemper, Andrew R. Winters, Hendrik Ranocha, Gregor J. Gassner (2020)
  A purely hyperbolic discontinuous Galerkin approach for self-gravitating gas dynamics
  [arXiv: 2008.10593](https://arxiv.org/abs/2008.10593)
in combination with [`initial_condition_eoc_test_coupled_euler_gravity`](@ref).
"""
@inline function source_terms_eoc_test_coupled_euler_gravity(u, x, t, equations::CompressibleEulerEquations2D)
  # Same settings as in `initial_condition_eoc_test_coupled_euler_gravity`
  c = 2.0
  A = 0.1
  G = 1.0 # gravitational constant, must match coupling solver
  C_grav = -2.0 * G / pi # 2 == 4 / ndims

  x1, x2 = x
  si, co = sincos(pi * (x1 + x2 - t))
  rhox = A * pi * co
  rho  = c + A *  si

  du1 = rhox
  du2 = rhox
  du3 = rhox
  du4 = (1.0 - C_grav*rho)*rhox

  return SVector(du1, du2, du3, du4)
end

"""
    source_terms_eoc_test_euler(u, x, t, equations::CompressibleEulerEquations2D)

Setup used for convergence tests of the Euler equations with self-gravity used in
- Michael Schlottke-Lakemper, Andrew R. Winters, Hendrik Ranocha, Gregor J. Gassner (2020)
  A purely hyperbolic discontinuous Galerkin approach for self-gravitating gas dynamics
  [arXiv: 2008.10593](https://arxiv.org/abs/2008.10593)
in combination with [`initial_condition_eoc_test_coupled_euler_gravity`](@ref).
"""
@inline function source_terms_eoc_test_euler(u, x, t, equations::CompressibleEulerEquations2D)
  # Same settings as in `initial_condition_eoc_test_coupled_euler_gravity`
  c = 2.0
  A = 0.1
  G = 1.0
  C_grav = -2 * G / pi # 2 == 4 / ndims

  x1, x2 = x
  si, co = sincos(pi * (x1 + x2 - t))
  rhox = A * pi * co
  rho  = c + A *  si

  du1 = rhox
  du2 = rhox * (1 -     C_grav * rho)
  du3 = rhox * (1 -     C_grav * rho)
  du4 = rhox * (1 - 3 * C_grav * rho)

  return SVector(du1, du2, du3, du4)
end


"""
    initial_condition_sedov_self_gravity(x, t, equations::CompressibleEulerEquations2D)

Adaptation of the Sedov blast wave with self-gravity taken from
- Michael Schlottke-Lakemper, Andrew R. Winters, Hendrik Ranocha, Gregor J. Gassner (2020)
  A purely hyperbolic discontinuous Galerkin approach for self-gravitating gas dynamics
  [arXiv: 2008.10593](https://arxiv.org/abs/2008.10593)
based on
- http://flash.uchicago.edu/site/flashcode/user_support/flash4_ug_4p62/node184.html#SECTION010114000000000000000
Should be used together with [`boundary_condition_sedov_self_gravity`](@ref).
"""
function initial_condition_sedov_self_gravity(x, t, equations::CompressibleEulerEquations2D)
  # Set up polar coordinates
  r = sqrt(x[1]^2 + x[2]^2)

  # Setup based on http://flash.uchicago.edu/site/flashcode/user_support/flash4_ug_4p62/node184.html#SECTION010114000000000000000
  r0 = 0.125 # = 4.0 * smallest dx (for domain length=8 and max-ref=8)
  E = 1.0
  p_inner   = (equations.gamma - 1) * E / (pi * r0^2)
  p_ambient = 1e-5 # = true Sedov setup

  # Calculate primitive variables
  # use a logistic function to tranfer density value smoothly
  L  = 1.0    # maximum of function
  x0 = 1.0    # center point of function
  k  = -150.0 # sharpness of transfer
  logistic_function_rho = L/(1.0 + exp(-k*(r - x0)))
  rho_ambient = 1e-5
  rho = max(logistic_function_rho, rho_ambient) # clip background density to not be so tiny

  # velocities are zero
  v1 = 0.0
  v2 = 0.0

  # use a logistic function to tranfer pressure value smoothly
  logistic_function_p = p_inner/(1.0 + exp(-k*(r - r0)))
  p = max(logistic_function_p, p_ambient)

  return prim2cons(SVector(rho, v1, v2, p), equations)
end

"""
    boundary_condition_sedov_self_gravity(u_inner, orientation, direction, x, t,
                                          surface_flux_function,
                                          equations::CompressibleEulerEquations2D)

Adaptation of the Sedov blast wave with self-gravity taken from
- Michael Schlottke-Lakemper, Andrew R. Winters, Hendrik Ranocha, Gregor J. Gassner (2020)
  A purely hyperbolic discontinuous Galerkin approach for self-gravitating gas dynamics
  [arXiv: 2008.10593](https://arxiv.org/abs/2008.10593)
based on
- http://flash.uchicago.edu/site/flashcode/user_support/flash4_ug_4p62/node184.html#SECTION010114000000000000000
Should be used together with [`initial_condition_sedov_self_gravity`](@ref).
"""
function boundary_condition_sedov_self_gravity(u_inner, orientation, direction, x, t,
                                               surface_flux_function,
                                               equations::CompressibleEulerEquations2D)
  # velocities are zero, density/pressure are ambient values according to
  # initial_condition_sedov_self_gravity
  rho = 1e-5
  v1 = 0.0
  v2 = 0.0
  p = 1e-5

  u_boundary = prim2cons(SVector(rho, v1, v2, p), equations)

  # Calculate boundary flux
  if direction in (2, 4) # u_inner is "left" of boundary, u_boundary is "right" of boundary
    flux = surface_flux_function(u_inner, u_boundary, orientation, equations)
  else # u_boundary is "left" of boundary, u_inner is "right" of boundary
    flux = surface_flux_function(u_boundary, u_inner, orientation, equations)
  end

  return flux
end


"""
    boundary_state_slip_wall(u_internal, normal_direction::AbstractVector,
                             equations::CompressibleEulerEquations2D)

Determine the external solution value for a slip wall condition. Sets the normal
velocity of the the exterior fictitious element to the negative of the internal value.

!!! warning "Experimental code"
    This wall function can change any time.
"""
@inline function boundary_state_slip_wall(u_internal, normal_direction::AbstractVector,
                                          equations::CompressibleEulerEquations2D)

  # normalize the outward pointing direction
  normal = normal_direction / norm(normal_direction)

  # compute the normal and tangential components of the velocity
  u_normal  = normal[1] * u_internal[2] + normal[2] * u_internal[3]
  u_tangent = (u_internal[2] - u_normal * normal[1], u_internal[3] - u_normal * normal[2])

  return SVector(u_internal[1],
                 u_tangent[1] - u_normal * normal[1],
                 u_tangent[2] - u_normal * normal[2],
                 u_internal[4])
end


# Calculate 1D flux for a single point
@inline function flux(u, orientation::Integer, equations::CompressibleEulerEquations2D)
  rho, rho_v1, rho_v2, rho_e = u
  v1 = rho_v1/rho
  v2 = rho_v2/rho
  p = (equations.gamma - 1) * (rho_e - 1/2 * rho * (v1^2 + v2^2))
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


# Called from the general surface flux routine in `numerical_fluxes.jl` so the direction
# has been normalized before this call
@inline function flux(u, normal_vector::AbstractVector, equations::CompressibleEulerEquations2D)
  rho, rho_v1, rho_v2, rho_e = u
  v1 = rho_v1/rho
  v2 = rho_v2/rho
  p = (equations.gamma - 1) * (rho_e - 1/2 * rho * (v1^2 + v2^2))

  v_normal = v1 * normal_vector[1] + v2 * normal_vector[2]
  rho_v_normal = rho * v_normal
  f1 = rho_v_normal
  f2 = rho_v_normal * v1 + p * normal_vector[1]
  f3 = rho_v_normal * v2 + p * normal_vector[2]
  f4 = (rho_e + p) * v_normal
  return SVector(f1, f2, f3, f4)
end


"""
    function flux_shima_etal(u_ll, u_rr, orientation, equations::CompressibleEulerEquations2D)

This flux is is a modification of the original kinetic energy preserving two-point flux by
- Kuya, Totani and Kawai (2018)
  Kinetic energy and entropy preserving schemes for compressible flows
  by split convective forms
  [DOI: 10.1016/j.jcp.2018.08.058](https://doi.org/10.1016/j.jcp.2018.08.058)
The modification is in the energy flux to guarantee pressure equilibrium and was developed by
- Nao Shima, Yuichi Kuya, Yoshiharu Tamaki, Soshi Kawai (JCP 2020)
  Preventing spurious pressure oscillations in split convective form discretizations for
  compressible flows
"""
@inline function flux_shima_etal(u_ll, u_rr, orientation::Integer, equations::CompressibleEulerEquations2D)
  # Unpack left and right state
  rho_ll, rho_v1_ll, rho_v2_ll, rho_e_ll = u_ll
  rho_rr, rho_v1_rr, rho_v2_rr, rho_e_rr = u_rr

  v1_ll = rho_v1_ll / rho_ll
  v2_ll = rho_v2_ll / rho_ll
  v1_rr = rho_v1_rr / rho_rr
  v2_rr = rho_v2_rr / rho_rr
  p_ll  = (equations.gamma - 1) * (rho_e_ll - 1/2 * rho_ll * (v1_ll^2 + v2_ll^2))
  p_rr  = (equations.gamma - 1) * (rho_e_rr - 1/2 * rho_rr * (v1_rr^2 + v2_rr^2))

  # Average each factor of products in flux
  rho_avg = 1/2 * (rho_ll + rho_rr)
  v1_avg  = 1/2 * ( v1_ll +  v1_rr)
  v2_avg  = 1/2 * ( v2_ll +  v2_rr)
  p_avg   = 1/2 * (  p_ll +   p_rr)
  kin_avg = 1/2 * (v1_ll*v1_rr + v2_ll*v2_rr)

  # Calculate fluxes depending on orientation
  if orientation == 1
    pv1_avg = 1/2 * (p_ll*v1_rr + p_rr*v1_ll)
    f1 = rho_avg * v1_avg
    f2 = rho_avg * v1_avg * v1_avg + p_avg
    f3 = rho_avg * v1_avg * v2_avg
    f4 = p_avg*v1_avg/(equations.gamma-1) + rho_avg*v1_avg*kin_avg + pv1_avg
  else
    pv2_avg = 1/2 * (p_ll*v2_rr + p_rr*v2_ll)
    f1 = rho_avg * v2_avg
    f2 = rho_avg * v2_avg * v1_avg
    f3 = rho_avg * v2_avg * v2_avg + p_avg
    f4 = p_avg*v2_avg/(equations.gamma-1) + rho_avg*v2_avg*kin_avg + pv2_avg
  end

  return SVector(f1, f2, f3, f4)
end


"""
    flux_kennedy_gruber(u_ll, u_rr, orientation, equations::CompressibleEulerEquations2D)

Kinetic energy preserving two-point flux by
- Kennedy and Gruber (2008)
  Reduced aliasing formulations of the convective terms within the
  Navier-Stokes equations for a compressible fluid
  [DOI: 10.1016/j.jcp.2007.09.020](https://doi.org/10.1016/j.jcp.2007.09.020)
"""
@inline function flux_kennedy_gruber(u_ll, u_rr, orientation::Integer, equations::CompressibleEulerEquations2D)
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
  p_avg = 1/2 * ((equations.gamma - 1) * (rho_e_ll - 1/2 * rho_ll * (v1_ll^2 + v2_ll^2)) +
                 (equations.gamma - 1) * (rho_e_rr - 1/2 * rho_rr * (v1_rr^2 + v2_rr^2)))
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
    flux_chandrashekar(u_ll, u_rr, orientation, equations::CompressibleEulerEquations2D)

Entropy conserving two-point flux by
- Chandrashekar (2013)
  Kinetic Energy Preserving and Entropy Stable Finite Volume Schemes
  for Compressible Euler and Navier-Stokes Equations
  [DOI: 10.4208/cicp.170712.010313a](https://doi.org/10.4208/cicp.170712.010313a)
"""
@inline function flux_chandrashekar(u_ll, u_rr, orientation::Integer, equations::CompressibleEulerEquations2D)
  # Unpack left and right state
  rho_ll, rho_v1_ll, rho_v2_ll, rho_e_ll = u_ll
  rho_rr, rho_v1_rr, rho_v2_rr, rho_e_rr = u_rr

  v1_ll = rho_v1_ll/rho_ll
  v2_ll = rho_v2_ll/rho_ll
  v1_rr = rho_v1_rr/rho_rr
  v2_rr = rho_v2_rr/rho_rr
  p_ll =  (equations.gamma - 1) * (rho_e_ll - 1/2 * rho_ll * (v1_ll^2 + v2_ll^2))
  p_rr =  (equations.gamma - 1) * (rho_e_rr - 1/2 * rho_rr * (v1_rr^2 + v2_rr^2))
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
    f4 = f1 * 0.5*(1/(equations.gamma-1)/beta_mean - velocity_square_avg)+f2*v1_avg + f3*v2_avg
  else
    f1 = rho_mean * v2_avg
    f2 = f1 * v1_avg
    f3 = f1 * v2_avg + p_mean
    f4 = f1 * 0.5*(1/(equations.gamma-1)/beta_mean - velocity_square_avg)+f2*v1_avg + f3*v2_avg
  end

  return SVector(f1, f2, f3, f4)
end


"""
    flux_ranocha(u_ll, u_rr, orientation, equations::CompressibleEulerEquations2D)

Entropy conserving and kinetic energy preserving two-point flux by
- Ranocha (2018)
  Generalised Summation-by-Parts Operators and Entropy Stability of Numerical Methods
  for Hyperbolic Balance Laws
  [PhD thesis, TU Braunschweig](https://cuvillier.de/en/shop/publications/7743)
See also
- Ranocha (2020)
  Entropy Conserving and Kinetic Energy Preserving Numerical Methods for
  the Euler Equations Using Summation-by-Parts Operators
  [Proceedings of ICOSAHOM 2018](https://doi.org/10.1007/978-3-030-39647-3_42)
"""
@inline function flux_ranocha(u_ll, u_rr, orientation::Integer, equations::CompressibleEulerEquations2D)
  # Unpack left and right state
  rho_ll, rho_v1_ll, rho_v2_ll, rho_e_ll = u_ll
  rho_rr, rho_v1_rr, rho_v2_rr, rho_e_rr = u_rr

  v1_ll = rho_v1_ll / rho_ll
  v2_ll = rho_v2_ll / rho_ll
  v1_rr = rho_v1_rr / rho_rr
  v2_rr = rho_v2_rr / rho_rr
  p_ll =  (equations.gamma - 1) * (rho_e_ll - 0.5 * rho_ll * (v1_ll^2 + v2_ll^2))
  p_rr =  (equations.gamma - 1) * (rho_e_rr - 0.5 * rho_rr * (v1_rr^2 + v2_rr^2))

  # Compute the necessary mean values
  rho_mean   = ln_mean(rho_ll, rho_rr)
  rho_p_mean = ln_mean(rho_ll / p_ll, rho_rr / p_rr)
  v1_avg = 0.5 * (v1_ll + v1_rr)
  v2_avg = 0.5 * (v2_ll + v2_rr)
  p_avg  = 0.5 * (p_ll + p_rr)
  velocity_square_avg = 0.5 * (v1_ll*v1_rr + v2_ll*v2_rr)

  # Calculate fluxes depending on orientation
  if orientation == 1
    f1 = rho_mean * v1_avg
    f2 = f1 * v1_avg + p_avg
    f3 = f1 * v2_avg
    f4 = f1 * ( velocity_square_avg + 1 / ((equations.gamma-1) * rho_p_mean) ) + 0.5 * (p_ll*v1_rr + p_rr*v1_ll)
  else
    f1 = rho_mean * v2_avg
    f2 = f1 * v1_avg
    f3 = f1 * v2_avg + p_avg
    f4 = f1 * ( velocity_square_avg + 1 / ((equations.gamma-1) * rho_p_mean) ) + 0.5 * (p_ll*v2_rr + p_rr*v2_ll)
  end

  return SVector(f1, f2, f3, f4)
end


# Calculate maximum wave speed for local Lax-Friedrichs-type dissipation as the
# maximum velocity magnitude plus the maximum speed of sound
# TODO: This doesn't really use the `orientation` - should it?
@inline function max_abs_speed_naive(u_ll, u_rr, orientation::Integer, equations::CompressibleEulerEquations2D)
  rho_ll, rho_v1_ll, rho_v2_ll, rho_e_ll = u_ll
  rho_rr, rho_v1_rr, rho_v2_rr, rho_e_rr = u_rr

  # Calculate primitive variables and speed of sound
  v1_ll = rho_v1_ll / rho_ll
  v2_ll = rho_v2_ll / rho_ll
  v_mag_ll = sqrt(v1_ll^2 + v2_ll^2)
  p_ll = (equations.gamma - 1) * (rho_e_ll - 1/2 * rho_ll * v_mag_ll^2)
  c_ll = sqrt(equations.gamma * p_ll / rho_ll)
  v1_rr = rho_v1_rr / rho_rr
  v2_rr = rho_v2_rr / rho_rr
  v_mag_rr = sqrt(v1_rr^2 + v2_rr^2)
  p_rr = (equations.gamma - 1) * (rho_e_rr - 1/2 * rho_rr * v_mag_rr^2)
  c_rr = sqrt(equations.gamma * p_rr / rho_rr)

  λ_max = max(v_mag_ll, v_mag_rr) + max(c_ll, c_rr)
end


@inline function max_abs_speed_naive(u_ll, u_rr, normal_direction::AbstractVector, equations::CompressibleEulerEquations2D)
  return max_abs_speed_naive(u_ll, u_rr, 0, equations) * norm(normal_direction)
end


# Calculate minimum and maximum wave speeds for HLL-type fluxes
@inline function min_max_speed_naive(u_ll, u_rr, orientation::Integer, equations::CompressibleEulerEquations2D)
  rho_ll, rho_v1_ll, rho_v2_ll, rho_e_ll = u_ll
  rho_rr, rho_v1_rr, rho_v2_rr, rho_e_rr = u_rr

  # Calculate primitive variables and speed of sound
  v1_ll = rho_v1_ll / rho_ll
  v2_ll = rho_v2_ll / rho_ll
  p_ll = (equations.gamma - 1) * (rho_e_ll - 1/2 * rho_ll * (v1_ll^2 + v2_ll^2))

  v1_rr = rho_v1_rr / rho_rr
  v2_rr = rho_v2_rr / rho_rr
  p_rr = (equations.gamma - 1) * (rho_e_rr - 1/2 * rho_rr * (v1_rr^2 + v2_rr^2))

  if orientation == 1 # x-direction
    λ_min = v1_ll - sqrt(equations.gamma * p_ll / rho_ll)
    λ_max = v1_rr + sqrt(equations.gamma * p_rr / rho_rr)
  else # y-direction
    λ_min = v2_ll - sqrt(equations.gamma * p_ll / rho_ll)
    λ_max = v2_rr + sqrt(equations.gamma * p_rr / rho_rr)
  end

  return λ_min, λ_max
end


@inline function min_max_speed_naive(u_ll, u_rr, normal_direction::AbstractVector,
                                     equations::CompressibleEulerEquations2D)
  rho_ll, rho_v1_ll, rho_v2_ll, rho_e_ll = u_ll
  rho_rr, rho_v1_rr, rho_v2_rr, rho_e_rr = u_rr

  # Calculate primitive variables and speed of sound
  v1_ll = rho_v1_ll / rho_ll
  v2_ll = rho_v2_ll / rho_ll
  p_ll = (equations.gamma - 1) * (rho_e_ll - 1/2 * rho_ll * (v1_ll^2 + v2_ll^2))

  v1_rr = rho_v1_rr / rho_rr
  v2_rr = rho_v2_rr / rho_rr
  p_rr = (equations.gamma - 1) * (rho_e_rr - 1/2 * rho_rr * (v1_rr^2 + v2_rr^2))

  v_normal_ll = v1_ll * normal_direction[1] + v2_ll * normal_direction[2]
  v_normal_rr = v1_rr * normal_direction[1] + v2_rr * normal_direction[2]

  λ_min = ( v_normal_ll - sqrt(equations.gamma * p_ll / rho_ll) ) * norm(normal_direction)
  λ_max = ( v_normal_rr + sqrt(equations.gamma * p_rr / rho_rr) ) * norm(normal_direction)

  return λ_min, λ_max
end


# Called inside `FluxRotated` in `numerical_fluxes.jl` so the direction
# has been normalized prior to this rotation of the state vector
@inline function rotate_to_x(u, normal_vector, equations::CompressibleEulerEquations2D)
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
@inline function rotate_from_x(u, normal_vector, equations::CompressibleEulerEquations2D)
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


 """
    flux_hllc(u_ll, u_rr, orientation, equations::CompressibleEulerEquations2D)

Computes the HLLC flux (HLL with Contact) for compressible Euler equations developed by E.F. Toro
[Lecture slides](http://www.prague-sum.com/download/2012/Toro_2-HLLC-RiemannSolver.pdf)
Signal speeds: [DOI: 10.1137/S1064827593260140](https://doi.org/10.1137/S1064827593260140)
"""
function flux_hllc(u_ll, u_rr, orientation::Integer, equations::CompressibleEulerEquations2D)
  # Calculate primitive variables and speed of sound
  rho_ll, rho_v1_ll, rho_v2_ll, rho_e_ll = u_ll
  rho_rr, rho_v1_rr, rho_v2_rr, rho_e_rr = u_rr

  v1_ll = rho_v1_ll / rho_ll
  v2_ll = rho_v2_ll / rho_ll
  e_ll  = rho_e_ll / rho_ll
  p_ll = (equations.gamma - 1) * (rho_e_ll - 1/2 * rho_ll * (v1_ll^2 + v2_ll^2))
  c_ll = sqrt(equations.gamma*p_ll/rho_ll)

  v1_rr = rho_v1_rr / rho_rr
  v2_rr = rho_v2_rr / rho_rr
  e_rr  = rho_e_rr / rho_rr
  p_rr = (equations.gamma - 1) * (rho_e_rr - 1/2 * rho_rr * (v1_rr^2 + v2_rr^2))
  c_rr = sqrt(equations.gamma*p_rr/rho_rr)

  # Obtain left and right fluxes
  f_ll = flux(u_ll, orientation, equations)
  f_rr = flux(u_rr, orientation, equations)

  # Compute Roe averages
  sqrt_rho_ll = sqrt(rho_ll)
  sqrt_rho_rr = sqrt(rho_rr)
  sum_sqrt_rho = sqrt_rho_ll + sqrt_rho_rr
  if orientation == 1 # x-direction
    vel_L = v1_ll
    vel_R = v1_rr
    ekin_roe = (sqrt_rho_ll * v2_ll + sqrt_rho_rr * v2_rr)^2
  elseif orientation == 2 # y-direction
    vel_L = v2_ll
    vel_R = v2_rr
    ekin_roe = (sqrt_rho_ll * v1_ll + sqrt_rho_rr * v1_rr)^2
  end
  vel_roe = (sqrt_rho_ll * vel_L + sqrt_rho_rr * vel_R) / sum_sqrt_rho
  ekin_roe = 0.5 * (vel_roe^2 + ekin_roe / sum_sqrt_rho^2)
  H_ll = (rho_e_ll + p_ll) / rho_ll
  H_rr = (rho_e_rr + p_rr) / rho_rr
  H_roe = (sqrt_rho_ll * H_ll + sqrt_rho_rr * H_rr) / sum_sqrt_rho
  c_roe = sqrt((equations.gamma - 1) * (H_roe - ekin_roe))
  Ssl = min(vel_L - c_ll, vel_roe - c_roe)
  Ssr = max(vel_R + c_rr, vel_roe + c_roe)
  sMu_L = Ssl - vel_L
  sMu_R = Ssr - vel_R

  if Ssl >= 0.0
    f1 = f_ll[1]
    f2 = f_ll[2]
    f3 = f_ll[3]
    f4 = f_ll[4]
  elseif Ssr <= 0.0
    f1 = f_rr[1]
    f2 = f_rr[2]
    f3 = f_rr[3]
    f4 = f_rr[4]
  else
    SStar = (p_rr - p_ll + rho_ll*vel_L*sMu_L - rho_rr*vel_R*sMu_R) / (rho_ll*sMu_L - rho_rr*sMu_R)
    if Ssl <= 0.0 <= SStar
      densStar = rho_ll*sMu_L / (Ssl-SStar)
      enerStar = e_ll + (SStar - vel_L) * (SStar + p_ll / (rho_ll * sMu_L))
      UStar1 = densStar
      UStar4 = densStar*enerStar
      if orientation == 1 # x-direction
        UStar2 = densStar*SStar
        UStar3 = densStar*v2_ll
      elseif orientation == 2 # y-direction
        UStar2 = densStar*v1_ll
        UStar3 = densStar*SStar
      end
      f1 = f_ll[1]+Ssl*(UStar1 - rho_ll)
      f2 = f_ll[2]+Ssl*(UStar2 - rho_v1_ll)
      f3 = f_ll[3]+Ssl*(UStar3 - rho_v2_ll)
      f4 = f_ll[4]+Ssl*(UStar4 - rho_e_ll)
    else
      densStar = rho_rr*sMu_R / (Ssr-SStar)
      enerStar = e_rr + (SStar - vel_R) * (SStar + p_rr / (rho_rr * sMu_R))
      UStar1 = densStar
      UStar4 = densStar*enerStar
      if orientation == 1 # x-direction
        UStar2 = densStar*SStar
        UStar3 = densStar*v2_rr
      elseif orientation == 2 # y-direction
        UStar2 = densStar*v1_rr
        UStar3 = densStar*SStar
      end
      f1 = f_rr[1]+Ssr*(UStar1 - rho_rr)
      f2 = f_rr[2]+Ssr*(UStar2 - rho_v1_rr)
      f3 = f_rr[3]+Ssr*(UStar3 - rho_v2_rr)
      f4 = f_rr[4]+Ssr*(UStar4 - rho_e_rr)
    end
  end
  return SVector(f1, f2, f3, f4)
end



@inline function max_abs_speeds(u, equations::CompressibleEulerEquations2D)
  rho, rho_v1, rho_v2, rho_e = u
  v1 = rho_v1 / rho
  v2 = rho_v2 / rho
  p = (equations.gamma - 1) * (rho_e - 1/2 * rho * (v1^2 + v2^2))
  c = sqrt(equations.gamma * p / rho)

  return abs(v1) + c, abs(v2) + c
end


# Convert conservative variables to primitive
@inline function cons2prim(u, equations::CompressibleEulerEquations2D)
  rho, rho_v1, rho_v2, rho_e = u

  v1 = rho_v1 / rho
  v2 = rho_v2 / rho
  p = (equations.gamma - 1) * (rho_e - 0.5 * rho * (v1^2 + v2^2))

  return SVector(rho, v1, v2, p)
end


# Convert conservative variables to entropy
@inline function cons2entropy(u, equations::CompressibleEulerEquations2D)
  rho, rho_v1, rho_v2, rho_e = u

  v1 = rho_v1 / rho
  v2 = rho_v2 / rho
  v_square = v1^2 + v2^2
  p = (equations.gamma - 1) * (rho_e - 0.5 * rho * v_square)
  s = log(p) - equations.gamma*log(rho)
  rho_p = rho / p

  w1 = (equations.gamma - s) / (equations.gamma-1) - 0.5 * rho_p * v_square
  w2 = rho_p * v1
  w3 = rho_p * v2
  w4 = -rho_p

  return SVector(w1, w2, w3, w4)
end

@inline function entropy2cons(w, equations::CompressibleEulerEquations2D)
  # See Hughes, Franca, Mallet (1986) A new finite element formulation for CFD
  # [DOI: 10.1016/0045-7825(86)90127-1](https://doi.org/10.1016/0045-7825(86)90127-1)
  @unpack gamma = equations

  # convert to entropy `-rho * s` used by Hughes, France, Mallet (1986)
  # instead of `-rho * s / (gamma - 1)`
  V1, V2, V3, V5 = w .* (gamma-1)

  # s = specific entropy, eq. (53)
  s = gamma - V1 + (V2^2 + V3^2)/(2*V5)

  # eq. (52)
  rho_iota = ((gamma-1) / (-V5)^gamma)^(1/(gamma-1))*exp(-s/(gamma-1))

  # eq. (51)
  rho      = -rho_iota * V5
  rho_v1   =  rho_iota * V2
  rho_v2   =  rho_iota * V3
  rho_e    =  rho_iota * (1-(V2^2 + V3^2)/(2*V5))
  return SVector(rho, rho_v1, rho_v2, rho_e)
end




# Convert primitive to conservative variables
@inline function prim2cons(prim, equations::CompressibleEulerEquations2D)
  rho, v1, v2, p = prim
  rho_v1 = rho * v1
  rho_v2 = rho * v2
  rho_e  = p/(equations.gamma-1) + 0.5 * (rho_v1 * v1 + rho_v2 * v2)
  return SVector(rho, rho_v1, rho_v2, rho_e)
end


@inline function density(u, equations::CompressibleEulerEquations2D)
 rho = u[1]
 return rho
end


@inline function pressure(u, equations::CompressibleEulerEquations2D)
 rho, rho_v1, rho_v2, rho_e = u
 p = (equations.gamma - 1) * (rho_e - 0.5 * (rho_v1^2 + rho_v2^2) / rho)
 return p
end


@inline function density_pressure(u, equations::CompressibleEulerEquations2D)
 rho, rho_v1, rho_v2, rho_e = u
 rho_times_p = (equations.gamma - 1) * (rho * rho_e - 0.5 * (rho_v1^2 + rho_v2^2))
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
@inline function entropy_thermodynamic(cons, equations::CompressibleEulerEquations2D)
  # Pressure
  p = (equations.gamma - 1) * (cons[4] - 1/2 * (cons[2]^2 + cons[3]^2) / cons[1])

  # Thermodynamic entropy
  s = log(p) - equations.gamma*log(cons[1])

  return s
end


# Calculate mathematical entropy for a conservative state `cons`
@inline function entropy_math(cons, equations::CompressibleEulerEquations2D)
  # Mathematical entropy
  S = -entropy_thermodynamic(cons, equations) * cons[1] / (equations.gamma - 1)

  return S
end


# Default entropy is the mathematical entropy
@inline entropy(cons, equations::CompressibleEulerEquations2D) = entropy_math(cons, equations)


# Calculate total energy for a conservative state `cons`
@inline energy_total(cons, ::CompressibleEulerEquations2D) = cons[4]


# Calculate kinetic energy for a conservative state `cons`
@inline function energy_kinetic(u, equations::CompressibleEulerEquations2D)
  rho, rho_v1, rho_v2, rho_e = u
  return (rho_v1^2 + rho_v2^2) / (2 * rho)
end


# Calculate internal energy for a conservative state `cons`
@inline function energy_internal(cons, equations::CompressibleEulerEquations2D)
  return energy_total(cons, equations) - energy_kinetic(cons, equations)
end
