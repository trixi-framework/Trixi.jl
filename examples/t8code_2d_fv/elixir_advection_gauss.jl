#!/usr/bin/env julia
# Has to be at least Julia v1.9.0.

# Running in parallel:
#   ${JULIA_DEPOT_PATH}/.julia/bin/mpiexecjl --project=. -n 3 julia hybrid-t8code-mesh.jl
#
# More information: https://juliaparallel.org/MPI.jl/stable/usage/
using OrdinaryDiffEq
using Trixi

using MPI
using T8code
using T8code.Libt8: SC_LP_ESSENTIAL
using T8code.Libt8: SC_LP_PRODUCTION

#
# Initialization.
#

# Initialize MPI. This has to happen before we initialize sc or t8code.
# mpiret = MPI.Init()

# We will use MPI_COMM_WORLD as a communicator.
comm = MPI.COMM_WORLD

# Initialize the sc library, has to happen before we initialize t8code.
T8code.Libt8.sc_init(comm, 1, 1, C_NULL, SC_LP_ESSENTIAL)
# Initialize t8code with log level SC_LP_PRODUCTION. See sc.h for more info on the log levels.
t8_init(SC_LP_PRODUCTION)

####################################################

advection_velocity = (0.2, -0.7)
equations = LinearScalarAdvectionEquation2D(advection_velocity)

initial_condition = initial_condition_gauss

solver = FV(order = 1, surface_flux = flux_lax_friedrichs)

initial_refinement_level = 4
mesh = T8codeFVMesh{2}(Trixi.cmesh_new_periodic_hybrid, initial_refinement_level)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)

ode = semidiscretize(semi, (0.0, 20.0));

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval=analysis_interval)

alive_callback = AliveCallback(analysis_interval=analysis_interval)

save_solution = SaveSolutionCallback(interval=1,
                                     solution_variables=cons2prim)

stepsize_callback = StepsizeCallback(cfl=0.5)

callbacks = CallbackSet(summary_callback, save_solution, analysis_callback, alive_callback, stepsize_callback)


###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false),#Euler(),
            dt=5.0e-2, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep=false, callback=callbacks);
summary_callback()

#
# Clean-up
#

t8_forest_unref(Ref(mesh.forest))
T8code.Libt8.sc_finalize()
# MPI.Finalize()
