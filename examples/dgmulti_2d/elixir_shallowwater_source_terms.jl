using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the shallow water equations

equations = ShallowWaterEquations2D(gravity_constant=9.81)

initial_condition = initial_condition_convergence_test

volume_flux = (flux_wintermeyer_etal, flux_nonconservative_wintermeyer_etal)
surface_flux = (flux_lax_friedrichs, flux_nonconservative_fjordholm_etal)
dg = DGMulti(polydeg=3, element_type = Quad(), approximation_type = SBP(),
             surface_integral = SurfaceIntegralWeakForm(surface_flux),
             volume_integral = VolumeIntegralFluxDifferencing(volume_flux))

cells_per_dimension = (8, 8)
mesh = DGMultiMesh(dg, cells_per_dimension,
                   coordinates_min=(0.0, 0.0), coordinates_max=(sqrt(2), sqrt(2)),
                   periodicity=true)


semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, dg;
                                    source_terms=source_terms_convergence_test)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 0.4)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval=analysis_interval, uEltype=real(dg))
alive_callback = AliveCallback(analysis_interval=analysis_interval)

callbacks = CallbackSet(summary_callback,
                        analysis_callback,
                        alive_callback)

###############################################################################
# run the simulation

sol = solve(ode, RDPK3SpFSAL49(); abstol=1.0e-7, reltol=1.0e-7,
            ode_default_options()..., callback=callbacks);

summary_callback() # print the timer summary
