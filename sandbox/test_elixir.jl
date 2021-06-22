using StartUpDG, StructArrays
using Trixi, OrdinaryDiffEq
using Plots

rd = RefElemData(Tri(), N=4)

dg = DG(rd, (), SurfaceIntegralWeakForm(FluxLaxFriedrichs()), VolumeIntegralWeakForm())
# dg = DG(rd, (), SurfaceIntegralWeakForm(FluxHLL()), VolumeIntegralWeakForm())

equations = CompressibleEulerEquations2D(1.4)

initial_condition = initial_condition_convergence_test
source_terms = source_terms_convergence_test

unif_mesh = StartUpDG.uniform_mesh(rd.elementType, 8)

# mesh = VertexMappedMesh(unif_mesh..., rd, is_periodic=(true,true))
# semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, dg,
#                                     source_terms = source_terms) 

top_boundary(x,y,tol=50*eps()) = abs(y-1)<tol 
rest_of_boundary(x,y,tol=50*eps()) = !top_boundary(x,y,tol)
is_on_boundary = Dict(:top => top_boundary, :rest => rest_of_boundary)
mesh = VertexMappedMesh(unif_mesh...,rd, is_on_boundary = is_on_boundary)
boundary_condition_convergence_test = BoundaryConditionDirichlet(initial_condition)
boundary_conditions = (; :top => boundary_condition_convergence_test,
                        :rest => boundary_condition_convergence_test)
semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, dg,
                                    source_terms = source_terms, 
                                    boundary_conditions = boundary_conditions) 

tspan = (0.0, .25)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()
alive_callback = AliveCallback(alive_interval=10)
callbacks = CallbackSet(summary_callback, alive_callback)

###############################################################################
# run the simulation

dt0 = StartUpDG.estimate_h(rd,mesh.md) / StartUpDG.inverse_trace_constant(rd)
sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false),
            dt = .5*dt0, save_everystep=false, callback=callbacks);
summary_callback() # print the timer summary

u = sol.u[end]
@show Trixi.calc_error_norms(nothing,u,tspan[end],nothing,mesh,equations,initial_condition,dg,semi.cache,nothing)

