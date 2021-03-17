using StartUpDG
using OrdinaryDiffEq
using Trixi
using UnPack

###############################################################################
# semidiscretization of the linear advection equation

N = 3
K1D = 16
CFL = .5
FinalTime = 2.0

VX,VY,EToV = uniform_mesh(Tri(),K1D)
rd = RefElemData(Tri(),N)
md = MeshData(VX,VY,EToV,rd)
md = make_periodic(md,rd)

struct JesseMesh
    VX
    VY    
    EToV    
end

Base.real(rd::RefElemData) = Float64

u0(x,y) = exp(-25*(x^2+y^2))

function Trixi.create_cache(mesh::JesseMesh, equations, rd::RefElemData, RealT, uEltype)

    @unpack VX,VY,EToV = mesh
    md = MeshData(VX,VY,EToV,rd)
    md = make_periodic(md,rd)

    cache = (;md)

    return cache
end

function Trixi.rhs!(du, u, t,
                    mesh::JesseMesh, equations,
                    initial_condition, boundary_conditions, source_terms,
                    rd::RefElemData, cache)

    @unpack md = cache
    @unpack rxJ,sxJ,J,nxJ,mapP = md
    @unpack Dr,Ds,LIFT,Vf = rd

    uf = Vf*u
    uflux = .5*(uf[mapP]-uf)
    rhsuJ = rxJ.*(Dr*u) + sxJ.*(Ds*u) + LIFT*(nxJ.*uflux)
    du .= -rhsuJ./J
    return nothing
end

################## interface stuff #################

Trixi.ndims(mesh::JesseMesh) = 2

Trixi.allocate_coefficients(mesh::JesseMesh, equations, rd::RefElemData, cache) = similar(cache.md.x)
function Trixi.compute_coefficients!(u, u0, t, mesh::JesseMesh, equations, rd::RefElemData, cache) 
    u .=  u0.(cache.md.x,cache.md.y)
end

Trixi.wrap_array(u_ode::Array{Float64,2}, semi::Trixi.AbstractSemidiscretization) = u_ode
Trixi.wrap_array(u_ode::Array{Float64,2}, mesh::JesseMesh, equations, solver, cache) = u_ode

Trixi.ndofs(mesh::JesseMesh, rd::RefElemData, cache) = length(rd.r)*cache.md.K

# A semidiscretization collects data structures and functions for the spatial discretization
semi = SemidiscretizationHyperbolic(JesseMesh(VX,VY,EToV), LinearScalarAdvectionEquation2D(1.0,0.0), u0, rd)

###############################################################################
# ODE solvers, callbacks etc.

# Create ODE problem with time span from 0.0 to 1.0
ode = semidiscretize(semi, (0.0, 1.0));

# At the beginning of the main loop, the SummaryCallback prints a summary of the simulation setup
# and resets the timers
summary_callback = SummaryCallback()

# # The AnalysisCallback allows to analyse the solution in regular intervals and prints the results
# analysis_callback = AnalysisCallback(semi, interval=100)

# # The SaveSolutionCallback allows to save the solution to a file in regular intervals
# save_solution = SaveSolutionCallback(interval=100,
#                                      solution_variables=cons2prim)

# # The StepsizeCallback handles the re-calculcation of the maximum Δt after each time step
# stepsize_callback = StepsizeCallback(cfl=1.6)

# Create a CallbackSet to collect all callbacks such that they can be passed to the ODE solver
# callbacks = CallbackSet(summary_callback, analysis_callback, save_solution, stepsize_callback)
callbacks = CallbackSet(summary_callback)


###############################################################################
# run the simulation

# OrdinaryDiffEq's `solve` method evolves the solution in time and executes the passed callbacks
sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false),
            dt = .5*md.J[1], # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep=false, callback=callbacks);

# Print the timer summary
summary_callback()
