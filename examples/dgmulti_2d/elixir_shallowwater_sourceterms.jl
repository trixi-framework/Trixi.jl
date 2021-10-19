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

cells_per_dimension = 8
vertex_coordinates, EToV = StartUpDG.uniform_mesh(dg.basis.elementType, cells_per_dimension)
vertex_coordinates = map(x -> sqrt(2) .* 0.5 .* (1 .+ x), vertex_coordinates) # map domain to [0, sqrt(2)]^2
mesh = VertexMappedMesh(vertex_coordinates, EToV, dg, is_periodic=(true, true))

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
stepsize_callback = StepsizeCallback(cfl=1.0)

callbacks = CallbackSet(summary_callback,
                        analysis_callback,
                        alive_callback,
                        stepsize_callback)

###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false),
            dt=1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep=false, callback=callbacks);

summary_callback() # print the timer summary
