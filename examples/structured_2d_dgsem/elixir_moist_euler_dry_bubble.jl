
using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the compressible moist Euler equations

equations = CompressibleMoistEulerEquations2D()

# Warm bubble test from paper:
# Wicker, L. J., and W. C. Skamarock, 1998: A time-splitting scheme
# for the elastic equations incorporating second-order Runge–Kutta
# time differencing. Mon. Wea. Rev., 126, 1992–1999.
function initial_condition_warm_bubble(x, t, equations::CompressibleMoistEulerEquations2D)
  @unpack p_0, kappa, g, c_pd, c_vd, R_d, R_v = equations
  xc = 10000.0
  zc = 2000.0
  r = sqrt((x[1] - xc)^2 + (x[2] - zc)^2)
  rc = 2000.0
  θ_ref = 300.0
  Δθ = 0.0

  if r <= rc
     Δθ = 2 * cospi(0.5*r/rc)^2
  end

  #Perturbed state:
  θ = θ_ref + Δθ # potential temperature
  # π_exner = 1 - g / (c_pd * θ) * x[2] # exner pressure
  # rho = p_0 / (R_d * θ) * (π_exner)^(c_vd / R_d) # density
  
  # calculate background pressure with assumption hydrostatic and neutral
  p = p_0 * (1-kappa * g * x[2] / (R_d * θ_ref))^(c_pd / R_d)
  
  #calculate rho and T with p and theta (now perturbed) rho = p / R_d T, T = θ / π
  rho = p / ((p / p_0)^kappa*R_d*θ)
  T = p / (R_d * rho)

  v1 = 20.0
  #v1 = 0.0
  v2 = 0.0
  rho_v1 = rho * v1
  rho_v2 = rho * v2
  rho_E = rho * c_vd * T + 1/2 * rho * (v1^2 + v2^2)  
  return SVector(rho, rho_v1, rho_v2, rho_E, zero(eltype(g)) ,zero(eltype(g)))
end

initial_condition = initial_condition_warm_bubble

boundary_condition = (x_neg=boundary_condition_periodic,
                      x_pos=boundary_condition_periodic,
                      y_neg=boundary_condition_slip_wall,
                      y_pos=boundary_condition_slip_wall)

# Gravity source since Q_ph=0
source_term = source_terms_geopotential

###############################################################################
# Get the DG approximation space
polydeg = 4
basis = LobattoLegendreBasis(polydeg)

surface_flux = flux_LMARS
volume_flux = flux_chandrashekar

volume_integral=VolumeIntegralFluxDifferencing(volume_flux)


# Create DG solver with polynomial degree = 4 and LMARS flux as surface flux 
# and the EC flux (chandrashekar) as volume flux
solver = DGSEM(basis, surface_flux, volume_integral)

coordinates_min = (0.0, 0.0)
coordinates_max = (20000.0, 10000.0)


cells_per_dimension = (64, 32)

# Create curved mesh with 64 x 32 elements
mesh = StructuredMesh(cells_per_dimension, coordinates_min, coordinates_max)

# A semidiscretization collects data structures and functions for the spatial discretization
semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                    boundary_conditions=boundary_condition,
                                    source_terms=source_term)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 1000.0)


# Create ODE problem with time span from 0.0 to 1000.0
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 1000
solution_variables = cons2drypot

analysis_callback = AnalysisCallback(semi, interval=analysis_interval, extra_analysis_errors=(:entropy_conservation_error, ))

alive_callback = AliveCallback(analysis_interval=analysis_interval)

save_solution = SaveSolutionCallback(interval=1000,
                                     save_initial_solution=true,
                                     save_final_solution=true,
                                     solution_variables=solution_variables)


stepsize_callback = StepsizeCallback(cfl=0.2)

callbacks = CallbackSet(summary_callback,
                        analysis_callback,
                        alive_callback,
                        save_solution,
                        stepsize_callback)


###############################################################################
# run the simulation
sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false), 
            maxiters=1.0e7,
            dt=1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep=false, callback=callbacks);
summary_callback() # print the timer summary
