using OrdinaryDiffEq, Trixi

using StructArrays
using RecursiveArrayTools,ArrayInterface
using LazyArrays

using LinearAlgebra,BenchmarkTools,Plots,UnPack

using Octavian,CheapThreads
using StartUpDG

include("trixi_interface.jl") 
include("flux_differencing.jl")
include("ModalESDG.jl")
include("modal_esdg.jl")

###############################################################################
# semidiscretization 

N = 3
K1D = 32
CFL = .1
FinalTime = .7

rd = RefElemData(Line(),N)
VX,EToV = uniform_mesh(Line(),K1D)

eqn = CompressibleEulerEquations1D(1.4)
function LxF_dissipation(u_ll, u_rr, orientation, equations::CompressibleEulerEquations1D)
    return .5 * max_abs_speed_naive(u_ll, u_rr, orientation, equations) * (u_ll-u_rr)
end
solver = ModalESDG(rd,Trixi.flux_chandrashekar,Trixi.flux_chandrashekar,LxF_dissipation,eqn)

function initial_condition(xyz,t,equations::CompressibleEulerEquations1D)
    x, = xyz
    ρ = 1 + .98*sin(pi*x)
    u = 1.0
    p = 2.
    return prim2cons((ρ,u,p),equations)
end

###############################################################################
# ODE solvers, callbacks etc.

# A semidiscretization collects data structures and functions for the spatial discretization
semi = SemidiscretizationHyperbolic(UnstructuredMesh((VX,),EToV), 
                                    CompressibleEulerEquations1D(1.4), 
                                    initial_condition, solver)

# Create ODE problem with time span from 0.0 to 1.0
ode = semidiscretize(semi, (0.0, FinalTime));

# At the beginning of the main loop, the SummaryCallback prints a summary of the simulation setup
# and resets the timers
summary_callback = SummaryCallback()
callbacks = CallbackSet(summary_callback)

###############################################################################
# run the simulation
CN = (N+1)*(N+2)/2
dt0 = CFL * 2. / (K1D*CN)

# OrdinaryDiffEq's `solve` method evolves the solution in time and executes the passed callbacks
# sol = solve(ode, SSPRK43(), dt=.01*dt0, save_everystep=false, callback=callbacks);
sol = solve(ode, Tsit5(), dt = dt0, save_everystep=false, callback=callbacks)

# Print the timer summary
summary_callback()

md = semi.cache.md
Q = sol.u[end]
zz = vec(rd.Vp*StructArrays.component(Q,1))
scatter(vec(rd.Vp*md.x),zz,leg=false,msw=0,ms=2,ratio=1)
