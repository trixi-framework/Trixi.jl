using StartUpDG, StructArrays
using Trixi, OrdinaryDiffEq
using Plots

rd = RefElemData(Tri(), N=4) # equivalent to a "basis"
dg = DG(rd, (), SurfaceIntegralWeakForm(FluxHLL()), VolumeIntegralWeakForm())

equations = CompressibleEulerEquations2D(1.4)
initial_condition = initial_condition_convergence_test
source_terms = source_terms_convergence_test

VX, VY, EToV = StartUpDG.uniform_mesh(rd.elementType, 8)
mesh = VertexMappedMesh(VX, VY, EToV, rd, is_periodic=(true,true))
semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, dg,
                                    source_terms = source_terms) 

tspan = (0.0, .1)
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
@show Trixi.calc_error_norms(cons2cons,u,tspan[end],nothing,mesh,equations,initial_condition,dg,semi.cache,nothing)

