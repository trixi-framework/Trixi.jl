using StartUpDG, StructArrays
using Trixi, OrdinaryDiffEq
using Plots

rd = RefElemData(Tri(),N=3)
K1D = 8
unif_mesh = StartUpDG.uniform_mesh(rd.elementType,K1D)

mesh = VertexMappedMesh(unif_mesh...,rd,is_periodic=(true,true))
dg = DG(rd,(),SurfaceIntegralWeakForm(FluxLaxFriedrichs()),VolumeIntegralWeakForm())
equations = CompressibleEulerEquations2D(1.4)

initial_condition = initial_condition_convergence_test
source_terms = source_terms_convergence_test

# top_boundary(x,y,tol=50*eps()) = abs(y-1)<tol 
# rest_of_boundary(x,y,tol=50*eps()) = !top_boundary(x,y,tol)
# is_on_boundary = Dict(:top => top_boundary, :rest => rest_of_boundary)
# mesh = VertexMappedMesh(unif_mesh...,rd,is_on_boundary=is_on_boundary)
# boundary_condition_convergence_test = BoundaryStateDirichlet(initial_condition)
# boundary_conditions = (; :top => boundary_condition_convergence_test,
#                         :rest => boundary_condition_convergence_test)
# semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, dg,
#                                     source_terms = source_terms, 
#                                     boundary_conditions = boundary_conditions) 

# equations = LinearScalarAdvectionEquation2D((1.0,1.0))
# function initial_condition(xyz,t,equations::LinearScalarAdvectionEquation2D)
#     return SVector{1}(1.0)
# end

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, dg,
                                    source_terms = source_terms) 

tspan = (0.0, .50)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()
alive_callback = AliveCallback(alive_interval=10)
callbacks = CallbackSet(summary_callback, alive_callback)

###############################################################################
# run the simulation

dt0 = StartUpDG.estimate_h(rd,mesh.md) / StartUpDG.inverse_trace_constant(rd)
sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false),
            dt = .75*dt0, save_everystep=false, callback=callbacks);
summary_callback() # print the timer summary

# md = mesh.md
# u = StructArrays.component(sol.u[end],1)
# uplot = rd.Vp*u
# scatter(vec.((x->rd.Vp*x).(md.xyz))..., vec(uplot), zcolor=vec(uplot), msw=0, leg=false, cam=(0,90))


u = ode.u0
du = similar(u)
Trixi.calc_surface_integral!(du, u, dg.surface_integral,mesh,equations,dg,semi.cache)
# Trixi.rhs!(du, ode.u0, semi, t)
