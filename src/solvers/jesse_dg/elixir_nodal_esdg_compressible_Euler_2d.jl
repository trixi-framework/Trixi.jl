using OrdinaryDiffEq, Trixi

using StructArrays, StaticArrays
using RecursiveArrayTools, ArrayInterface
using LazyArrays
using LinearAlgebra, Plots, BenchmarkTools, UnPack

using Octavian, CheapThreads
using LoopVectorization

using StartUpDG

include("trixi_interface.jl") # some setup utils
include("flux_differencing.jl")
include("NodalESDG.jl")
include("nodal_esdg.jl")

###############################################################################
# semidiscretization

N = 3
K1D = 16
CFL = .1
FinalTime = .75

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
solver = NodalESDG(N,Tri(),Trixi.flux_chandrashekar,Trixi.flux_chandrashekar,LxF_dissipation,eqn)

element_type = Tri()
# VX,VY,EToV = uniform_mesh(element_type,K1D)

h = .1
triin=Triangulate.TriangulateIO()
triin.pointlist=Matrix{Cdouble}([-1.0 -1.0;
                                  1.0 -1.0;
                                  1.0  1.0;
                                 -1.0  1.0;
                                 -0.1 -0.1;
                                  0.1 -0.1;
                                  0.1  0.1;
                                 -0.1  0.1;
                                 ]')
triin.segmentlist=Matrix{Cint}([1 2; 2 3; 3 4; 4 1; 5 6; 6 7; 7 8; 8 5; ]')
triin.segmentmarkerlist=Vector{Int32}([1, 1,1,1, 2,2,2,2])
triin.holelist=[0. 0.]'
triout = triangulate(triin,h^2)
VX,VY,EToV = triangulateIO_to_VXYZEToV(triout)

md = MeshData(VX,VY,EToV,solver.rd)
# md = make_periodic(md,solver.rd)

function initial_condition(xyz,t,equations::CompressibleEulerEquations2D)
    x,y = xyz
    # ϱ = 1 + .5*exp(-25*(x^2+y^2))
    # u = 1.0
    # v = .5
    # p = 2.

    ϱ = 1 + .5*exp(-100*((x-.5)^2+(y-.5)^2))
    u = 0.0
    v = 0.0
    p = ϱ^equations.gamma
    return prim2cons((ϱ,u,v,p),equations)
end

# A semidiscretization collects data structures and functions for the spatial discretization
semi = SemidiscretizationHyperbolic(md, eqn, initial_condition, solver)

###############################################################################
# ODE solvers, callbacks etc.

# Create ODE problem with time span from 0.0 to 1.0
ode = semidiscretize(semi, (0.0, FinalTime));

# At the beginning of the main loop, the SummaryCallback prints a summary of the simulation setup
# and resets the timers
summary_callback = SummaryCallback()
callbacks = CallbackSet(summary_callback)

###############################################################################
# run the simulation
CN = (N+1)*(N+2)/2
dt0 = CFL * 2 / (K1D*CN)

# OrdinaryDiffEq's `solve` method evolves the solution in time and executes the passed callbacks
# sol = solve(ode, SSPRK43(), dt=.01*dt0, save_everystep=false, callback=callbacks);
# sol = solve(ode, Tsit5(), dt = dt0, save_everystep=false, callback=callbacks)
sol = solve(ode,CarpenterKennedy2N54(williamson_condition=false), dt=dt0, save_everystep=false, callback=callbacks)

# Print the timer summary
summary_callback()

Q = sol.u[end]
rd = solver.rd
zz = vec(rd.Vp*rd.Pq*StructArrays.component(Q,1))
scatter(vec(rd.Vp*md.x),vec(rd.Vp*md.y),zz,zcolor=zz,leg=false,msw=0,ms=2,cam=(0,90),ratio=1)

# dQ = similar(Q)
# cache = semi.cache