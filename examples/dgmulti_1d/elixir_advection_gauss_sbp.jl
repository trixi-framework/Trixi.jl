
using OrdinaryDiffEq
using Trixi

###############################################################################
#  setup the equations

advection_velocity = 1.0
equations = LinearScalarAdvectionEquation1D(advection_velocity)

###############################################################################
#  setup the GSBP DG discretization that uses the Gauss operators from Chan et al.

surface_flux = FluxLaxFriedrichs()
dg = DGMulti(polydeg = 3,
             element_type = Line(),
             approximation_type = GaussSBP(),
             surface_integral = SurfaceIntegralWeakForm(surface_flux),
             volume_integral = VolumeIntegralWeakForm())

###############################################################################
#  setup the 1D mesh

cells_per_dimension = (8,)
mesh = DGMultiMesh(dg, cells_per_dimension,
                   coordinates_min=(-1.0,), coordinates_max=(1.0,),
                   periodicity=true)

###############################################################################
#  setup the test problem (no source term needed for linear advection)

initial_condition = initial_condition_convergence_test

###############################################################################
#  setup the semidiscretization and ODE problem

semi = SemidiscretizationHyperbolic(mesh,
                                    equations,
                                    initial_condition,
                                    dg)

tspan = (0.0, 1.5)
ode = semidiscretize(semi, tspan)

###############################################################################
#  setup the callbacks

# prints a summary of the simulation setup and resets the timers
summary_callback = SummaryCallback()

# analyse the solution in regular intervals and prints the results
analysis_callback = AnalysisCallback(semi, interval=100, uEltype=real(dg))

# handles the re-calculation of the maximum Δt after each time step
stepsize_callback = StepsizeCallback(cfl=0.75)

# collect all callbacks such that they can be passed to the ODE solver
callbacks = CallbackSet(summary_callback, analysis_callback, stepsize_callback)

###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false),
            dt=1.0, save_everystep=false, callback=callbacks);

# Print the timer summary
summary_callback()

