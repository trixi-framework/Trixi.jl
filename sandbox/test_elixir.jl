using StartUpDG, StructArrays
using Trixi, OrdinaryDiffEq
using Plots

rd = RefElemData(Tri(),N=3)
K1D = 8
unif_mesh = StartUpDG.uniform_mesh(rd.elementType,K1D)
mesh = VertexMappedMesh(unif_mesh...,rd,is_periodic=(true,true))

# top_boundary(x,y,tol=50*eps()) = abs(y-1)<tol 
# rest_of_boundary(x,y,tol=50*eps()) = !top_boundary(x,y,tol)
# boundary_conditions = Dict(:top => top_boundary, 
#                            :rest => rest_of_boundary)

dg = DG(rd,(),SurfaceIntegralWeakForm(FluxLaxFriedrichs()),VolumeIntegralWeakForm())
equations = CompressibleEulerEquations2D(1.4)

function initial_condition_sine(xyz,t,equations::CompressibleEulerEquations2D)
    x,y = xyz
    rho = 2 + .1*sin(pi*x)*sin(pi*y)
    u = .5
    v = .5
    p = 2.0
    return prim2cons(SVector{4}(rho,u,v,p),equations)
end
function source_terms_zero(u,xyz,t,equations::CompressibleEulerEquations2D)
    return SVector{4}(0.0,0.0,0.0,0.0)
end

initial_condition = initial_condition_sine
source_terms = source_terms_zero

# equations = LinearScalarAdvectionEquation2D((1.0,1.0))
# function initial_condition(xyz,t,equations::LinearScalarAdvectionEquation2D)
#     return SVector{1}(1.0)
# end

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, dg,
                                    source_terms = source_terms)

tspan = (0.0, .50)
ode = semidiscretize(semi, tspan)

# du = similar(ode.u0)
# Trixi.rhs!(du, ode.u0, semi, t)

summary_callback = SummaryCallback()
alive_callback = AliveCallback(alive_interval=10)
callbacks = CallbackSet(summary_callback, alive_callback)

###############################################################################
# run the simulation

dt0 = StartUpDG.estimate_h(rd,mesh.md) / StartUpDG.inverse_trace_constant(rd)
sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false),
            dt = .75*dt0, save_everystep=false, callback=callbacks);
summary_callback() # print the timer summary

md = mesh.md
u = StructArrays.component(sol.u[end],1)
uplot = rd.Vp*u
scatter(vec.((x->rd.Vp*x).(md.xyz))..., vec(uplot), zcolor=vec(uplot), msw=0, leg=false, cam=(0,90))