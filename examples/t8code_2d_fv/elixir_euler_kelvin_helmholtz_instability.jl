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


function cmesh_new_periodic_tri(comm, n_dims)::t8_cmesh_t
	vertices = [ # Just all vertices of all trees. partly duplicated
		-1.0, -1.0, 0, # tree 0, triangle
		1.0, -1.0, 0,
		1.0, 1.0, 0,
		-1.0, -1.0, 0, # tree 1, triangle
		1.0, 1.0, 0,
		-1.0, 1.0, 0,

        # -1.0, 0, 0,  # tree 0, triangle
		# 0, -1.0, 0,
		# 1.0, 0, 0,
		# -1.0, 0, 0,  # tree 1, triangle
		# 1.0, 0, 0,
		# 0, 1.0, 0,
	]

	# Generally, one can define other geometries. But besides linear the other
	# geometries in t8code do not have C interface yet.
	linear_geom = t8_geometry_linear_new(n_dims)

	#
	# This is how the cmesh looks like. The numbers are the tree numbers:
	#
	#   +---+
	#   |1 /|
	#   | / |
	#   |/0 |
	#   +---+
	#

	cmesh_ref = Ref(t8_cmesh_t())
	t8_cmesh_init(cmesh_ref)
	cmesh = cmesh_ref[]

	# /* Use linear geometry */
	t8_cmesh_register_geometry(cmesh, linear_geom)
	t8_cmesh_set_tree_class(cmesh, 0, T8_ECLASS_TRIANGLE)
	t8_cmesh_set_tree_class(cmesh, 1, T8_ECLASS_TRIANGLE)

    t8_cmesh_set_tree_vertices(cmesh, 0, @views(vertices[1 +  0:end]), 3)
    t8_cmesh_set_tree_vertices(cmesh, 1, @views(vertices[1 +  9:end]), 3)

	t8_cmesh_set_join(cmesh, 0, 1, 1, 2, 0)
	t8_cmesh_set_join(cmesh, 0, 1, 0, 1, 0)
	t8_cmesh_set_join(cmesh, 0, 1, 2, 0, 1)

	t8_cmesh_commit(cmesh, comm)

	return cmesh
end

function cmesh_new_periodic_tri2(comm, n_dims)::t8_cmesh_t
	vertices = [ # Just all vertices of all trees. partly duplicated
		-1.0, -1.0, 0, # tree 0, triangle
		0, -1.0, 0,
		0, 0, 0,
		-1.0, -1.0, 0, # tree 1, triangle
		0, 0, 0,
		-1.0, 0, 0,
        0, -1.0, 0, # tree 2, triangle
		1.0, -1.0, 0,
		1.0, 0, 0,
		0, -1.0, 0, # tree 3, triangle
		1.0, 0, 0,
		0, 0, 0,
        -1.0, 0, 0, # tree 4, triangle
		0, 0, 0,
		-1.0, 1.0, 0,
		-1.0, 1.0, 0, # tree 5, triangle
		0, 0, 0,
		0, 1.0, 0,
        0, 0, 0, # tree 6, triangle
		1.0, 0, 0,
		0, 1.0, 0,
		0, 1.0, 0, # tree 7, triangle
		1.0, 0, 0,
		1.0, 1.0, 0,
	]

	# Generally, one can define other geometries. But besides linear the other
	# geometries in t8code do not have C interface yet.
	linear_geom = t8_geometry_linear_new(n_dims)

	#
	# This is how the cmesh looks like. The numbers are the tree numbers:
	#
    #   +---+---+
    #   |\ 5|\ 7|
    #   | \ | \ |
    #   |4 \| 6\|
	#   +---+---+
	#   |1 /|3 /|
	#   | / | / |
	#   |/0 |/ 2|
	#   +---+---+
	#

	cmesh_ref = Ref(t8_cmesh_t())
	t8_cmesh_init(cmesh_ref)
	cmesh = cmesh_ref[]

	# /* Use linear geometry */
	t8_cmesh_register_geometry(cmesh, linear_geom)
	t8_cmesh_set_tree_class(cmesh, 0, T8_ECLASS_TRIANGLE)
	t8_cmesh_set_tree_class(cmesh, 1, T8_ECLASS_TRIANGLE)
    t8_cmesh_set_tree_class(cmesh, 2, T8_ECLASS_TRIANGLE)
	t8_cmesh_set_tree_class(cmesh, 3, T8_ECLASS_TRIANGLE)
    t8_cmesh_set_tree_class(cmesh, 4, T8_ECLASS_TRIANGLE)
	t8_cmesh_set_tree_class(cmesh, 5, T8_ECLASS_TRIANGLE)
    t8_cmesh_set_tree_class(cmesh, 6, T8_ECLASS_TRIANGLE)
	t8_cmesh_set_tree_class(cmesh, 7, T8_ECLASS_TRIANGLE)

    t8_cmesh_set_tree_vertices(cmesh, 0, @views(vertices[1 +  0:end]), 3)
    t8_cmesh_set_tree_vertices(cmesh, 1, @views(vertices[1 +  9:end]), 3)
    t8_cmesh_set_tree_vertices(cmesh, 2, @views(vertices[1 +  18:end]), 3)
    t8_cmesh_set_tree_vertices(cmesh, 3, @views(vertices[1 +  27:end]), 3)
    t8_cmesh_set_tree_vertices(cmesh, 4, @views(vertices[1 +  36:end]), 3)
    t8_cmesh_set_tree_vertices(cmesh, 5, @views(vertices[1 +  45:end]), 3)
    t8_cmesh_set_tree_vertices(cmesh, 6, @views(vertices[1 +  54:end]), 3)
    t8_cmesh_set_tree_vertices(cmesh, 7, @views(vertices[1 +  63:end]), 3)

	t8_cmesh_set_join(cmesh, 0, 1, 1, 2, 0)
	t8_cmesh_set_join(cmesh, 0, 3, 0, 1, 0)
	t8_cmesh_set_join(cmesh, 0, 5, 2, 1, 0)

    t8_cmesh_set_join(cmesh, 1, 4, 0, 2, 1)
    t8_cmesh_set_join(cmesh, 1, 2, 1, 0, 0)

    t8_cmesh_set_join(cmesh, 2, 3, 1, 2, 0)
    t8_cmesh_set_join(cmesh, 2, 7, 2, 1, 0)

    t8_cmesh_set_join(cmesh, 3, 6, 0, 2, 1)

    t8_cmesh_set_join(cmesh, 4, 5, 0, 2, 1)
    t8_cmesh_set_join(cmesh, 4, 7, 1, 0, 0)

    t8_cmesh_set_join(cmesh, 5, 6, 0, 1, 0)

    t8_cmesh_set_join(cmesh, 6, 7, 0, 2, 1)

	t8_cmesh_commit(cmesh, comm)

	return cmesh
end


function build_forest(comm, n_dims, level)
	# Periodic mesh of quads and triangles.
	# cmesh = t8_cmesh_new_periodic_hybrid(comm) # The `t8code` version does exactly the same.
	# cmesh = cmesh_new_periodic_hybrid(comm, n_dims)
	# cmesh = cmesh_new_periodic_quad(comm, n_dims)
	# cmesh = cmesh_new_periodic_tri(comm, n_dims)
	cmesh = cmesh_new_periodic_tri2(comm, n_dims)

	scheme = t8_scheme_new_default_cxx()

	let do_face_ghost = 1
		forest = t8_forest_new_uniform(cmesh, scheme, level, do_face_ghost, comm)
		return forest
	end
end


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

# Initialize an adapted forest with periodic boundaries.
n_dims = 2
initial_refinement_level = 4
forest = build_forest(comm, n_dims, initial_refinement_level)


number_trees = t8_forest_get_num_local_trees(forest)
println("rank $(Trixi.mpi_rank()): #trees $number_trees, #elements $(t8_forest_get_local_num_elements(forest)), #ghost_elements $(t8_forest_get_num_ghosts(forest))")

mesh = T8codeMesh{n_dims}(forest)

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

t8_forest_unref(Ref(forest))
T8code.Libt8.sc_finalize()
# MPI.Finalize()
