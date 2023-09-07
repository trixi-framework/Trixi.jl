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


###############################################################################
# semidiscretization of the compressible Euler equations
gamma = 1.4
equations = CompressibleEulerEquations2D(gamma)

"""
    initial_condition_kelvin_helmholtz_instability(x, t, equations::CompressibleEulerEquations2D)

A version of the classical Kelvin-Helmholtz instability based on
- Andrés M. Rueda-Ramírez, Gregor J. Gassner (2021)
  A Subcell Finite Volume Positivity-Preserving Limiter for DGSEM Discretizations
  of the Euler Equations
  [arXiv: 2102.06017](https://arxiv.org/abs/2102.06017)
"""
function initial_condition_kelvin_helmholtz_instability(x, t, equations::CompressibleEulerEquations2D)
  # change discontinuity to tanh
  # typical resolution 128^2, 256^2
  # domain size is [-1,+1]^2
  slope = 15
  amplitude = 0.02
  B = tanh(slope * x[2] + 7.5) - tanh(slope * x[2] - 7.5)
  rho = 0.5 + 0.75 * B
  v1 = 0.5 * (B - 1)
  v2 = 0.1 * sin(2 * pi * x[1])
  p = 1.0
  return prim2cons(SVector(rho, v1, v2, p), equations)
end
initial_condition = initial_condition_kelvin_helmholtz_instability

solver = FV(order = 2, slope_limiter = Trixi.monotonized_central, surface_flux = flux_lax_friedrichs)

initial_refinement_level = 4
# mesh_function = Trixi.cmesh_new_periodic_hybrid
# mesh_function = Trixi.cmesh_new_periodic_quad
# mesh_function = Trixi.cmesh_new_periodic_tri
mesh_function = Trixi.cmesh_new_periodic_tri2
mesh = T8codeMesh{2}(mesh_function, initial_refinement_level)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)

tspan = (0.0, 3.7)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval=analysis_interval)

alive_callback = AliveCallback(analysis_interval=analysis_interval)

save_solution = SaveSolutionCallback(interval=40,
                                     save_initial_solution=true,
                                     save_final_solution=true,
                                     solution_variables=cons2prim)

stepsize_callback = StepsizeCallback(cfl=0.9)

callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback,
                        save_solution,
                        stepsize_callback)


###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false),
            dt=1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep=false, callback=callbacks);
summary_callback() # print the timer summary

#
# Clean-up
#

t8_forest_unref(Ref(mesh.forest))
T8code.Libt8.sc_finalize()
# MPI.Finalize()
