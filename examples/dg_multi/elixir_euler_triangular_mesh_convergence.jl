# !!! warning "Experimental features"

# run using
# convergence_test(joinpath(examples_dir(), "triangular_mesh_2D", "elixir_euler_triangular_mesh_convergence.jl"), 4)

using Trixi, OrdinaryDiffEq

dg = DGMulti(; polydeg = 3, elem_type = Tri(),
               surface_integral = SurfaceIntegralWeakForm(FluxLaxFriedrichs()),
               volume_integral = VolumeIntegralWeakForm())

equations = CompressibleEulerEquations2D(1.4)
initial_condition = initial_condition_convergence_test
source_terms = source_terms_convergence_test

# example where we tag two separate boundary segments of the mesh
cells_per_dimension = (8,8) # detected by `extract_initial_resolution` for convergence tests
vertex_coordinates_x, vertex_coordinates_y, EToV = StartUpDG.uniform_mesh(Tri(), cells_per_dimension...)
mesh = VertexMappedMesh(vertex_coordinates_x, vertex_coordinates_y, EToV, dg)

boundary_condition_convergence_test = BoundaryConditionDirichlet(initial_condition)
boundary_conditions = (; :entire_boundary => boundary_condition_convergence_test)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, dg,
                                    source_terms = source_terms,
                                    boundary_conditions = boundary_conditions)

tspan = (0.0, 0.25)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()
alive_callback = AliveCallback(alive_interval=10)
analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval=analysis_interval, uEltype=real(dg))
callbacks = CallbackSet(summary_callback, alive_callback, analysis_callback)

###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false),
            dt = 0.5 * estimate_dt(dg, mesh), save_everystep=false, callback=callbacks);
summary_callback() # print the timer summary
