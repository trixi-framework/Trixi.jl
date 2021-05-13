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
md = MeshData(VX,EToV,rd)
md = make_periodic(md,rd)

eqn = CompressibleEulerEquations1D(1.4)
solver = ModalESDG(rd,Trixi.flux_chandrashekar,Trixi.flux_chandrashekar,
                   Trixi.DissipationLocalLaxFriedrichs(),
                   Trixi.cons2entropy,Trixi.entropy2cons,eqn)

function initial_condition(xyz,t,equations::CompressibleEulerEquations1D)
    x, = xyz
    ϱ = 1 + .98*sin(pi*x)
    u = 1.0
    p = 2.
    return prim2cons((ϱ,u,p),equations)
end

###############################################################################
# ODE solvers, callbacks etc.

# A semidiscretization collects data structures and functions for the spatial discretization
semi = SemidiscretizationHyperbolic(md, eqn, initial_condition, solver)

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

Q = sol.u[end]
zz = vec(rd.Vp*StructArrays.component(Q,1))
scatter(vec(rd.Vp*md.x),zz,leg=false,msw=0,ms=2,ratio=1)

# cache = semi.cache
# Q = Trixi.allocate_coefficients(md, eqn, solver, cache)
# dQ = similar(Q)
# Trixi.compute_coefficients!(Q, initial_condition, 0.0, md, eqn, solver, cache)
# Trixi.rhs!(dQ, Q, nothing, md, eqn, initial_condition, nothing, nothing, solver, cache)