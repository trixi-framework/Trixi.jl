# !!! warning "Experimental features"

using Trixi, OrdinaryDiffEq

flux_ec = flux_ranocha
dg = DGMulti(; polydeg = 3, elem_type = Tet(),
               surface_integral = SurfaceIntegralWeakForm(flux_ec),
               volume_integral = VolumeIntegralFluxDifferencing(flux_ec))


equations = CompressibleEulerEquations3D(1.4)
initial_condition = initial_condition_convergence_test
source_terms = source_terms_convergence_test

# example where we tag two separate boundary segments of the mesh
top_boundary(x, y, z, tol=50*eps()) = abs(z - 1) < tol
rest_of_boundary(x, y, z, tol=50*eps()) = !top_boundary(x, y, z, tol)
is_on_boundary = Dict(:top => top_boundary, :rest => rest_of_boundary)
vertex_coordinates_x, vertex_coordinates_y, vertex_coordinates_z, EToV = StartUpDG.uniform_mesh(Tet(), 2)
mesh = VertexMappedMesh((vertex_coordinates_x, vertex_coordinates_y, vertex_coordinates_z),
                        EToV, dg, is_on_boundary = is_on_boundary)

boundary_condition_convergence_test = BoundaryConditionDirichlet(initial_condition)
boundary_conditions = (; :top => boundary_condition_convergence_test,
                        :rest => boundary_condition_convergence_test)

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
