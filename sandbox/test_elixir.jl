using StartUpDG, StructArrays
using Trixi, OrdinaryDiffEq
using Plots

rd = RefElemData(Tri(),N=3)
K1D = 16
mesh = StartUpDG.uniform_mesh(rd.elementType,K1D)

dg = DG(rd,(),SurfaceIntegralWeakForm(FluxLaxFriedrichs()),VolumeIntegralWeakForm())
equations = CompressibleEulerEquations2D(1.4)
function initial_condition(xyz,t,equations::CompressibleEulerEquations2D)
    x,y = xyz
    rho = 2 + .1*sin(pi*x)*sin(pi*y)
    u = .5
    v = .5
    p = 2.0
    return prim2cons(SVector{4}(rho,u,v,p),equations)
end

function source_terms(u,xyz,t,equations::CompressibleEulerEquations2D)
    return SVector{4}(0.0,0.0,0.0,0.0)
end
# equations = LinearScalarAdvectionEquation2D((1.0,1.0))
# function initial_condition(xyz,t,equations::LinearScalarAdvectionEquation2D)
#     return SVector{1}(1.0)
# end

semi = SemidiscretizationHyperbolic(mesh,equations,initial_condition,dg,
                                    source_terms = source_terms,
                                    boundary_conditions=nothing)

tspan = (0.0, .50)
ode = semidiscretize(semi, tspan)

# du = similar(ode.u0)
# Trixi.rhs!(du, ode.u0, semi, t)

summary_callback = SummaryCallback()
alive_callback = AliveCallback(alive_interval=10)
callbacks = CallbackSet(summary_callback, alive_callback)

###############################################################################
# run the simulation

dt0 = StartUpDG.estimate_h(rd,semi.cache.md) / StartUpDG.inverse_trace_constant(rd)
sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false),
            dt= .5*dt0, save_everystep=false, callback=callbacks);
summary_callback() # print the timer summary

md = semi.cache.md
u = StructArrays.component(sol.u[end],1)
Plots.scatter(vec.(md.xyz)...,vec(u),zcolor=vec(u),leg=false,cam=(0,90))