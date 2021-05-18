using OrdinaryDiffEq, Trixi

using StructArrays, StaticArrays
using RecursiveArrayTools, ArrayInterface
using LazyArrays
using LinearAlgebra, Plots, BenchmarkTools, UnPack

using Octavian, CheapThreads
using LoopVectorization

using StartUpDG

include("triangulate_support.jl")
include("triangulate_examples.jl")
include("trixi_interface.jl") # some setup utils
include("flux_differencing.jl")
include("ModalESDG.jl")
include("modal_esdg.jl")

###############################################################################
# semidiscretization

N = 3
CFL = .1
FinalTime = 2.5

h = .1
element_type = Tri()
triout = scramjet(h)
VX,VY,EToV = triangulateIO_to_VXYEToV(triout)
rd = RefElemData(element_type,N)
md = MeshData(VX,VY,EToV,rd)
is_boundary_node = get_node_boundary_tags(triout,rd,md) 
# is_boundary_node[is_boundary_node .!= 0] .= 1

eqn = CompressibleEulerEquations2D(1.4)
@inline function max_abs_speed_normal(UL, UR, normal, equations::CompressibleEulerEquations2D)
    # Calculate primitive variables and speed of sound
    ρu_n_L = UL[2]*normal[1] + UL[3]*normal[2]
    ρu_n_R = UR[2]*normal[1] + UR[3]*normal[2]
    uL = (UL[1],ρu_n_L,UL[4])
    uR = (UR[1],ρu_n_R,UR[4])
    return Trixi.max_abs_speed_naive(uL,uR,0,CompressibleEulerEquations1D(equations.gamma))
end
@inline function LxF_dissipation(u_ll, u_rr, normal, equations::CompressibleEulerEquations2D)
    return .5 * max_abs_speed_normal(u_ll, u_rr, normal, equations) * (u_ll-u_rr)
end

solver = ModalESDG(rd,Trixi.flux_chandrashekar,Trixi.flux_chandrashekar,LxF_dissipation,
                   Trixi.cons2entropy,Trixi.entropy2cons,eqn)

function initial_condition(xyz,t,equations::CompressibleEulerEquations2D)
    x,y = xyz
    # ϱ = 1 + .5*exp(-25*(x^2+y^2))    
    # u = 1.0
    # v = .5
    # p = 2.

    # ϱ = 1 + .5*exp(-100*((x-1.5)^2+(y-1)^2))    
    # u = 0.0
    # v = 0.0
    # p = ϱ^equations.gamma

    γ = equations.gamma
    ϱ = γ
    u = 1.1
    v = 0.0
    p = 1.0 # a = sqrt(1.4*p/ϱ) = 1.0

    return prim2cons((ϱ,u,v,p),equations)
end

# A semidiscretization collects data structures and functions for the spatial discretization
semi = SemidiscretizationHyperbolic(md, CompressibleEulerEquations2D(1.4),
                                    initial_condition, solver; initial_cache=(;is_boundary_node))

###############################################################################
# ODE solvers, callbacks etc.

# Create ODE problem with time span from 0.0 to 1.0
ode = semidiscretize(semi, (0.0, FinalTime));

# At the beginning of the main loop, the SummaryCallback prints a summary of the simulation setup
# and resets the timers
summary_callback = SummaryCallback()
alive_callback = AliveCallback(alive_interval=100)
callbacks = CallbackSet(alive_callback,summary_callback)

###############################################################################
# run the simulation
CN = (N+1)*(N+2)/2
dt0 = CFL * 2 * h / CN

# OrdinaryDiffEq's `solve` method evolves the solution in time and executes the passed callbacks
sol = solve(ode, SSPRK43(), dt = dt0, save_everystep=false, callback=callbacks);
# sol = solve(ode, Tsit5(), dt = dt0, save_everystep=false, callback=callbacks)
# sol = solve(ode,CarpenterKennedy2N54(williamson_condition=false), dt=dt0, save_everystep=false, callback=callbacks)

# Print the timer summary
summary_callback()

Q = sol.u[end]
zz = vec(rd.Vp*StructArrays.component(Q,1))
scatter(vec(rd.Vp*md.x),vec(rd.Vp*md.y),zz,zcolor=zz,leg=false,msw=0,ms=1,cam=(0,90),ratio=1)

dQ = similar(Q)
cache = semi.cache
equations = CompressibleEulerEquations2D(1.4)