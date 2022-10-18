
using OrdinaryDiffEq
using Trixi

###############################################################################
# Semidiscretization of the shallow water equations

equations = ShallowWaterEquations1D(gravity_constant=9.81)

bottom_topography(x) = 2.0 + 0.5 * sin(sqrt(2.0) * pi * x)
coordinates_min = 0.0
coordinates_max = sqrt(2.0)
n = 100

interp_x = Vector(LinRange(coordinates_min, coordinates_max, n))
interp_y = bottom_topography.(interp_x)

# Spline interpolation
spline         = cubic_b_spline(interp_x, interp_y; end_condition="not-a-knot")
spline_func(x) = spline_interpolation(spline, x)

function initial_condition_convergence_test_spline(x, t, equations::ShallowWaterEquations1D)
  # some constants are chosen such that the function is periodic on the domain [0,sqrt(2)]
  c  = 7.0
  omega_x = 2.0 * pi * sqrt(2.0)
  omega_t = 2.0 * pi
  
  H = c + cos(omega_x * x[1]) * cos(omega_t * t)
  v = 0.5
  b = spline_func(x[1])
  return prim2cons(SVector(H, v, b), equations)
end

initial_condition = initial_condition_convergence_test_spline


###############################################################################
# Get the DG approximation space

volume_flux = (flux_wintermeyer_etal, flux_nonconservative_wintermeyer_etal)
solver = DGSEM(polydeg=3, surface_flux=(flux_lax_friedrichs, flux_nonconservative_fjordholm_etal),
               volume_integral=VolumeIntegralFluxDifferencing(volume_flux))

###############################################################################
# Get the TreeMesh and setup a periodic mesh

mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level=3,
                n_cells_max=10_000,
                periodicity=true)

# create the semi discretization object
semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                    source_terms=source_terms_convergence_test)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 1.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 500
analysis_callback = AnalysisCallback(semi, interval=analysis_interval)

alive_callback = AliveCallback(analysis_interval=analysis_interval)

save_solution = SaveSolutionCallback(interval=200,
                                     save_initial_solution=true,
                                     save_final_solution=true)

callbacks = CallbackSet(summary_callback, analysis_callback, alive_callback, save_solution)

###############################################################################
# run the simulation

# use a Runge-Kutta method with automatic (error based) time step size control
sol = solve(ode, RDPK3SpFSAL49(), abstol=1.0e-8, reltol=1.0e-8,
            save_everystep=false, callback=callbacks);
summary_callback() # print the timer summary
