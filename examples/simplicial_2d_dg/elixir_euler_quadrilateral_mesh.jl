# !!! warning "Experimental features"

using StartUpDG
using Trixi, OrdinaryDiffEq

polydeg = 3
rd = RefElemData(Quad(), polydeg)
dg = DG(rd, nothing #= mortar =#,
        SurfaceIntegralWeakForm(FluxHLL()), VolumeIntegralWeakForm())

equations = CompressibleEulerEquations2D(1.4)
initial_condition = initial_condition_convergence_test
source_terms = source_terms_convergence_test

# example where we tag two separate boundary segments of the mesh
top_boundary(x,y,tol=50*eps()) = abs(y-1)<tol
rest_of_boundary(x,y,tol=50*eps()) = !top_boundary(x,y,tol)
is_on_boundary = Dict(:top => top_boundary, :rest => rest_of_boundary)
vertex_coordinates_x, vertex_coordinates_y, EToV = StartUpDG.uniform_mesh(rd.elementType, 4)
mesh = VertexMappedMesh(vertex_coordinates_x, vertex_coordinates_y, EToV, rd, is_on_boundary = is_on_boundary)

boundary_condition_convergence_test = BoundaryConditionDirichlet(initial_condition)
boundary_conditions = (; :top => boundary_condition_convergence_test,
                        :rest => boundary_condition_convergence_test)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, dg,
                                    source_terms = source_terms,
                                    boundary_conditions = boundary_conditions)

tspan = (0.0, 0.4)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()
alive_callback = AliveCallback(alive_interval=10)
analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval=analysis_interval, uEltype=real(dg))
callbacks = CallbackSet(summary_callback, alive_callback, analysis_callback)

###############################################################################
# run the simulation

dt0 = StartUpDG.estimate_h(rd,mesh.md) / StartUpDG.inverse_trace_constant(rd)
sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false),
            dt = 0.5*dt0, save_everystep=false, callback=callbacks);
summary_callback() # print the timer summary
