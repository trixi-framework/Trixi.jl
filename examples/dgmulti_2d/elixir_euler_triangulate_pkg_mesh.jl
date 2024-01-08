using Trixi, OrdinaryDiffEq

dg = DGMulti(polydeg = 3, element_type = Tri(),
             surface_integral = SurfaceIntegralWeakForm(flux_hll),
             volume_integral = VolumeIntegralWeakForm())

equations = CompressibleEulerEquations2D(1.4)
initial_condition = initial_condition_convergence_test
source_terms = source_terms_convergence_test

# use pre-defined Triangulate geometry in StartUpDG
meshIO = StartUpDG.triangulate_domain(StartUpDG.RectangularDomainWithHole())

# the pre-defined Triangulate geometry in StartUpDG has integer boundary tags. this routine
# assigns boundary faces based on these integer boundary tags.
mesh = DGMultiMesh(dg, meshIO, Dict(:outer_boundary => 1, :inner_boundary => 2))

boundary_condition_convergence_test = BoundaryConditionDirichlet(initial_condition)
boundary_conditions = (; :outer_boundary => boundary_condition_convergence_test,
                       :inner_boundary => boundary_condition_convergence_test)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, dg,
                                    source_terms = source_terms,
                                    boundary_conditions = boundary_conditions)

tspan = (0.0, 0.2)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()
alive_callback = AliveCallback(alive_interval = 10)
analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval = analysis_interval, uEltype = real(dg))
callbacks = CallbackSet(summary_callback, alive_callback, analysis_callback)

###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false),
            dt = 0.5 * estimate_dt(mesh, dg), save_everystep = false, callback = callbacks);
summary_callback() # print the timer summary
