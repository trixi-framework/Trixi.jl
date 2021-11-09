# An idealized baroclinic instability test case
# For optimal results consider increasing the resolution to 16x16x8 trees per cube face.
#
# Note that this elixir can take several hours to run.
# Using 24 threads of an AMD Ryzen Threadripper 3990X (more threads don't speed it up further)
# and `check-bounds=no`, this elixirs takes about one hour to run.
# With 16x16x8 trees per cube face on the same machine, it takes about 28 hours.
#
# References:
# - Paul A. Ullrich, Thomas Melvin, Christiane Jablonowski, Andrew Staniforth (2013)
#   A proposed baroclinic wave test case for deep- and shallow-atmosphere dynamical cores
#   https://doi.org/10.1002/qj.2241

using OrdinaryDiffEq
using Trixi
using LinearAlgebra

###############################################################################
# semidiscretization of the compressible Euler equations
gamma = 1.4
equations = CompressibleEulerEquations3D(gamma)

# Initial condition for an idealized baroclinic instability test
# https://doi.org/10.1002/qj.2241, Section 3.2 and Appendix A
function initial_condition_baroclinic_instability(x, t, equations::CompressibleEulerEquations3D)
  # Parameters from Table 1 in the paper
  # Corresponding names in the paper are commented
  radius_earth          = 6.371229e6  # a
  half_width_parameter  = 2           # b
  gravity_constant      = 9.80616     # g
  k                     = 3           # k
  p0                    = 1e5         # p₀
  gas_constant          = 287         # R
  temperature0e         = 310.0       # T₀ᴱ
  temperature0p         = 240.0       # T₀ᴾ
  lapse                 = 0.005       # Γ
  omega_earth           = 7.29212e-5  # Ω


  lon, lat, r = cartesian_to_sphere(x)
  # Make sure that the r is not smaller than radius_earth
  z = max(r - radius_earth, 0.0)
  r = z + radius_earth

  # In the paper: T₀
  temperature0 = 0.5 * (temperature0e + temperature0p)
  # In the paper: A, B, C, H
  const_a = 1 / lapse
  const_b = (temperature0 - temperature0p) / (temperature0 * temperature0p)
  const_c = 0.5 * (k + 2) * (temperature0e - temperature0p) / (temperature0e * temperature0p)
  const_h = gas_constant * temperature0 / gravity_constant

  # In the paper: (r - a) / bH
  scaled_z = z / (half_width_parameter * const_h)

  # Temporary variables
  temp1 = exp(lapse/temperature0 * z)
  temp2 = exp(-scaled_z^2)

  # In the paper: ̃τ₁, ̃τ₂
  tau1 = const_a * lapse / temperature0 * temp1 + const_b * (1 - 2 * scaled_z^2) * temp2
  tau2 = const_c * (1 - 2 * scaled_z^2) * temp2

  # In the paper: ∫τ₁(r') dr', ∫τ₂(r') dr'
  inttau1 = const_a * (temp1 - 1) + const_b * z * temp2
  inttau2 = const_c * z * temp2

  # Temporary variables
  temp3 = r/radius_earth * cos(lat)
  temp4 = temp3^k - k/(k + 2) * temp3^(k+2)

  # In the paper: T
  temperature = 1 / ((r/radius_earth)^2 * (tau1 - tau2 * temp4))

  # In the paper: U, u (zonal wind, first component of spherical velocity)
  big_u = gravity_constant/radius_earth * k * temperature * inttau2 * (temp3^(k-1) - temp3^(k+1))
  temp5 = radius_earth * cos(lat)
  u = -omega_earth * temp5 + sqrt(omega_earth^2 * temp5^2 + temp5 * big_u)

  # Hydrostatic pressure
  p = p0 * exp(-gravity_constant/gas_constant * (inttau1 - inttau2 * temp4))

  # Perturbation
  u += evaluate_exponential(lon,lat,z)

  # Convert spherical velocity to Cartesian
  v1 = -sin(lon) * u
  v2 =  cos(lon) * u
  v3 = 0.0

  # Density (via ideal gas law)
  rho = p / (gas_constant * temperature)

  return prim2cons(SVector(rho, v1, v2, v3, p), equations)
end

function cartesian_to_sphere(x)
  r = norm(x)
  lambda = atan(x[2], x[1])
  if lambda < 0
    lambda += 2 * pi
  end
  phi = asin(x[3] / r)

  return lambda, phi, r
end

# Exponential perturbation function, taken from Fortran initialisation routine provided with
# https://doi.org/10.1002/qj.2241
function evaluate_exponential(lon, lat, z)
  pertup = 1.0
  pertexpr = 0.1
  pertlon = pi/9.0
  pertlat = 2.0*pi/9.0
  pertz = 15000.0

  # Great circle distance
  greatcircler = 1/pertexpr * acos(
      sin(pertlat) * sin(lat) + cos(pertlat) * cos(lat) * cos(lon - pertlon))

  # Vertical tapering of stream function
  if z < pertz
    perttaper = 1.0 - 3 * z^2 / pertz^2 + 2 * z^3 / pertz^3
  else
    perttaper = 0.0
  end

  # Zonal velocity perturbation
  if greatcircler < 1
    result = pertup * perttaper * exp(- greatcircler^2)
  else
    result = 0.0
  end

  return result
end


@inline function source_terms_baroclinic_instability(u, x, t, equations::CompressibleEulerEquations3D)
  radius_earth          = 6.371229e6  # a
  gravity_constant      = 9.80616     # g
  omega_earth           = 7.29212e-5  # Ω

  r = norm(x)
  # Make sure that the r is not smaller than radius_earth
  z = max(r - radius_earth, 0.0)
  r = z + radius_earth

  du1 = zero(eltype(u))

  # Gravity term
  temp = -gravity_constant * radius_earth^2 / r^3
  du2 = temp * u[1] * x[1]
  du3 = temp * u[1] * x[2]
  du4 = temp * u[1] * x[3]
  du5 = temp * (u[2] * x[1] + u[3] * x[2] + u[4] * x[3])

  # Coriolis term, -2Ω × ρv = -2 * omega_earth * (0, 0, 1) × u[2:4]
  du2 -= -2 * omega_earth * u[3]
  du3 -=  2 * omega_earth * u[2]

  return SVector(du1, du2, du3, du4, du5)
end


initial_condition = initial_condition_baroclinic_instability

boundary_conditions = Dict(
  :inside  => boundary_condition_slip_wall,
  :outside => boundary_condition_slip_wall,
)

# This is a good estimate for the speed of sound in this example.
# Other values between 300 and 400 should work as well.
surface_flux = FluxLMARS(340)
volume_flux  = flux_kennedy_gruber
solver = DGSEM(polydeg=5, surface_flux=surface_flux, volume_integral=VolumeIntegralFluxDifferencing(volume_flux))

# For optimal results, use (16, 8) here
trees_per_cube_face = (8, 4)
mesh = Trixi.P4estMeshCubedSphere(trees_per_cube_face..., 6.371229e6, 30000.0,
                                  polydeg=5, initial_refinement_level=0)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                    source_terms=source_terms_baroclinic_instability,
                                    boundary_conditions=boundary_conditions)


###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 10 * 24 * 60 * 60.0) # time in seconds for 10 days
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 5000
analysis_callback = AnalysisCallback(semi, interval=analysis_interval)

alive_callback = AliveCallback(analysis_interval=analysis_interval)

save_solution = SaveSolutionCallback(interval=5000,
                                     save_initial_solution=true,
                                     save_final_solution=true,
                                     solution_variables=cons2prim)

callbacks = CallbackSet(summary_callback,
                        analysis_callback,
                        alive_callback,
                        save_solution)


###############################################################################
# run the simulation

# Use a Runge-Kutta method with automatic (error based) time step size control
# Enable threading of the RK method for better performance on multiple threads
sol = solve(ode, RDPK3SpFSAL49(thread=OrdinaryDiffEq.True()), abstol=1.0e-6, reltol=1.0e-6,
            save_everystep=false, callback=callbacks);

summary_callback() # print the timer summary
