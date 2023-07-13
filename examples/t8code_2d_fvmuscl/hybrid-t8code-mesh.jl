#!/usr/bin/env julia
# Has to be at least Julia v1.9.0.

# Running in parallel:
#   ${JULIA_DEPOT_PATH}/.julia/bin/mpiexecjl --project=. -n 3 julia hybrid-t8code-mesh.jl
#
# More information: https://juliaparallel.org/MPI.jl/stable/usage/
# using OrdinaryDiffEq
using Trixi

using MPI
using T8code
# using T8code.Libt8: T8code.Libt8.sc_init
# using T8code.Libt8: T8code.Libt8.sc_free
# using T8code.Libt8: T8code.Libt8.sc_finalize
# using T8code.Libt8: T8code.Libt8.sc_array_new_data
# using T8code.Libt8: T8code.Libt8.sc_array_destroy
using T8code.Libt8: SC_LP_ESSENTIAL
using T8code.Libt8: SC_LP_PRODUCTION

# Directly ported from: `src/t8_cmesh/t8_cmesh_examples.c: t8_cmesh_new_periodic_hybrid`.
function cmesh_new_periodic_hybrid(comm, n_dims) :: t8_cmesh_t
  vertices = [  # /* Just all vertices of all trees. partly duplicated */
    0, 0, 0,                    # /* tree 0, triangle */
    0.5, 0, 0,
    0.5, 0.5, 0,
    0, 0, 0,                    # /* tree 1, triangle */
    0.5, 0.5, 0,
    0, 0.5, 0,
    0.5, 0, 0,                  # /* tree 2, quad */
    1, 0, 0,
    0.5, 0.5, 0,
    1, 0.5, 0,
    0, 0.5, 0,                  # /* tree 3, quad */
    0.5, 0.5, 0,
    0, 1, 0,
    0.5, 1, 0,
    0.5, 0.5, 0,                # /* tree 4, triangle */
    1, 0.5, 0,
    1, 1, 0,
    0.5, 0.5, 0,                # /* tree 5, triangle */
    1, 1, 0,
    0.5, 1, 0
  ]

  # Generally, one can define other geometries. But besides linear the other
  # geometries in t8code do not have C interface yet.
  linear_geom = t8_geometry_linear_new(n_dims)

  # /*
  #  *  This is how the cmesh looks like. The numbers are the tree numbers:
  #  *
  #  *   +---+---+
  #  *   |   |5 /|
  #  *   | 3 | / |
  #  *   |   |/ 4|
  #  *   +---+---+
  #  *   |1 /|   |
  #  *   | / | 2 |
  #  *   |/0 |   |
  #  *   +---+---+
  #  */

  cmesh_ref = Ref(t8_cmesh_t())
  t8_cmesh_init(cmesh_ref)
  cmesh = cmesh_ref[]

  # /* Use linear geometry */
  t8_cmesh_register_geometry(cmesh, linear_geom)
  t8_cmesh_set_tree_class(cmesh, 0, T8_ECLASS_TRIANGLE)
  t8_cmesh_set_tree_class(cmesh, 1, T8_ECLASS_TRIANGLE)
  t8_cmesh_set_tree_class(cmesh, 2, T8_ECLASS_QUAD)
  t8_cmesh_set_tree_class(cmesh, 3, T8_ECLASS_QUAD)
  t8_cmesh_set_tree_class(cmesh, 4, T8_ECLASS_TRIANGLE)
  t8_cmesh_set_tree_class(cmesh, 5, T8_ECLASS_TRIANGLE)

  t8_cmesh_set_tree_vertices(cmesh, 0, @views(vertices[1 +  0:end]), 3)
  t8_cmesh_set_tree_vertices(cmesh, 1, @views(vertices[1 +  9:end]), 3)
  t8_cmesh_set_tree_vertices(cmesh, 2, @views(vertices[1 + 18:end]), 4)
  t8_cmesh_set_tree_vertices(cmesh, 3, @views(vertices[1 + 30:end]), 4)
  t8_cmesh_set_tree_vertices(cmesh, 4, @views(vertices[1 + 42:end]), 3)
  t8_cmesh_set_tree_vertices(cmesh, 5, @views(vertices[1 + 51:end]), 3)

  t8_cmesh_set_join(cmesh, 0, 1, 1, 2, 0)
  t8_cmesh_set_join(cmesh, 0, 2, 0, 0, 0)
  t8_cmesh_set_join(cmesh, 0, 3, 2, 3, 0)

  t8_cmesh_set_join(cmesh, 1, 3, 0, 2, 1)
  t8_cmesh_set_join(cmesh, 1, 2, 1, 1, 0)

  t8_cmesh_set_join(cmesh, 2, 4, 3, 2, 0)
  t8_cmesh_set_join(cmesh, 2, 5, 2, 0, 1)

  t8_cmesh_set_join(cmesh, 3, 5, 1, 1, 0)
  t8_cmesh_set_join(cmesh, 3, 4, 0, 0, 0)

  t8_cmesh_set_join(cmesh, 4, 5, 1, 2, 0)

  t8_cmesh_commit(cmesh, comm)

  return cmesh
end

function cmesh_new_periodic_tri(comm, n_dims) :: t8_cmesh_t
    vertices = [  # /* Just all vertices of all trees. partly duplicated */
      0, 0, 0,                    # /* tree 0, triangle */
      1.0, 0, 0,
      1.0, 1.0, 0,
      0, 0, 0,                    # /* tree 1, triangle */
      1.0, 1.0, 0,
      0, 1.0, 0,
    ]

    # Generally, one can define other geometries. But besides linear the other
    # geometries in t8code do not have C interface yet.
    linear_geom = t8_geometry_linear_new(n_dims)

    # /*
    #  *  This is how the cmesh looks like. The numbers are the tree numbers:
    #  *
    #  *   +---+
    #  *   |1 /|
    #  *   | / |
    #  *   |/0 |
    #  *   +---+
    #  */

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
    t8_cmesh_set_join(cmesh, 0, 1, 2, 0, 0)

    t8_cmesh_commit(cmesh, comm)

    return cmesh
end

function cmesh_new_periodic_quad(comm, n_dims) :: t8_cmesh_t
    vertices = [  # /* Just all vertices of all trees. partly duplicated */
      0, 0, 0,                    # /* tree 0, quad */
      1.0, 0, 0,
      0, 1.0, 0,
      1.0, 1.0, 0
      ]

    # Generally, one can define other geometries. But besides linear the other
    # geometries in t8code do not have C interface yet.
    linear_geom = t8_geometry_linear_new(n_dims)

    # /*
    #  *  This is how the cmesh looks like. The numbers are the tree numbers:
    #  *
    #  *   +---+
    #  *   |   |
    #  *   | 0 |
    #  *   |   |
    #  *   +---+
    #  */

    cmesh_ref = Ref(t8_cmesh_t())
    t8_cmesh_init(cmesh_ref)
    cmesh = cmesh_ref[]

    # /* Use linear geometry */
    t8_cmesh_register_geometry(cmesh, linear_geom)
    t8_cmesh_set_tree_class(cmesh, 0, T8_ECLASS_QUAD)
    # t8_cmesh_set_tree_class(cmesh, 1, T8_ECLASS_QUAD)

    t8_cmesh_set_tree_vertices(cmesh, 0, @views(vertices[1 +  0:end]), 4)
    # t8_cmesh_set_tree_vertices(cmesh, 1, @views(vertices[1 +  12:end]), 4)

    t8_cmesh_set_join(cmesh, 0, 0, 0, 1, 0)
    t8_cmesh_set_join(cmesh, 0, 0, 2, 3, 0)

    t8_cmesh_commit(cmesh, comm)

    return cmesh
end

function build_forest(comm, n_dims, level, case)
    # More information and meshes: https://github.com/DLR-AMR/t8code/blob/main/src/t8_cmesh/t8_cmesh_examples.c#L1481
    if case == 1
        # Periodic mesh of quads.
        cmesh = cmesh_new_periodic_quad(comm, n_dims)
    elseif case == 2
        # Periodic mesh of triangles.
        cmesh = cmesh_new_periodic_tri(comm, n_dims)
    elseif case == 3
        # Periodic mesh of quads and triangles.
        # cmesh = t8_cmesh_new_periodic_hybrid(comm) # The `t8code` version does exactly the same.
        cmesh = cmesh_new_periodic_hybrid(comm, n_dims)
    else
        error("case = $case not allowed.")
    end
    scheme = t8_scheme_new_default_cxx()

    let do_face_ghost = 1
        forest = t8_forest_new_uniform(cmesh, scheme, level, do_face_ghost, comm)
        return forest
    end
end

# Write the forest as vtu and also write the element's volumes in the file.
#
# t8code supports writing element based data to vtu as long as its stored
# as doubles. Each of the data fields to write has to be provided in its own
# array of length num_local_elements.
# We support two types: T8_VTK_SCALAR - One double per element.
#                  and  T8_VTK_VECTOR - Three doubles per element.
function output_data_to_vtu(semi, u, out)
  vtk_data = [
    t8_vtk_data_field_t(
      T8_VTK_SCALAR,
      NTuple{8192, Cchar}(rpad("scalar\0", 8192, ' ')),
      pointer(u),
    ),
  ]

  # The number of user defined data fields to write.
  num_data = length(vtk_data)

  # Write user defined data to vtu file.
  write_treeid = 1
  write_mpirank = 1
  write_level = 1
  write_element_id = 1
  write_ghosts = 0
  t8_forest_write_vtk_ext(semi.mesh.forest, out, write_treeid, write_mpirank,
                           write_level, write_element_id, write_ghosts,
                           0, 0, num_data, pointer(vtk_data))
end

function fv_method_first_order(semi)
    mesh, equations, solver, cache = Trixi.mesh_equations_solver_cache(semi)
    @unpack elements = cache
    num_elements = length(elements)

    u = Array{typeof(elements[1].volume)}(undef, num_elements)
    for element in 1:num_elements
        u[element] = semi.initial_condition(elements[element].midpoint, 0.0, equations)
    end
    du = zeros(size(u))

    # Output the data to vtu files.
    prefix_forest_with_data = "hybrid_mesh_example"
    output_data_to_vtu(semi, u, prefix_forest_with_data * "_0")

    rhs!(du, u, semi)
    # u += du

    # Output the data to vtu files.
    # prefix_forest_with_data = "hybrid_mesh_example_t+1"
    # output_data_to_vtu(semi, u, prefix_forest_with_data * "_1")

    return nothing
end

function rhs!(du, u, semi)
    mesh, equations, solver, cache = Trixi.mesh_equations_solver_cache(semi)

    num_elements = t8_forest_get_local_num_elements(mesh.forest)

    for element in 1:num_elements
        @unpack volume, face_normals, num_faces, face_areas, face_connectivity = cache.elements[element]
        for face in 1:num_faces
            neighbor = face_connectivity[face]
            normal = @views([face_normals[2 * face - 1 : 2 * face]...]) # Unfortunaly, flux() requires an Vector and no Tuple
            du[element] += - #=(1 / volume) *=# face_areas[face] * solver.surface_flux(u[element], u[neighbor], normal, equations)
        end
        du[element] = (1 / volume) * du[element]
    end

    return nothing
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

# Initialize an adapted forest with periodic boundaries.
n_dims = 2
refinement_level = 4
# cases: 1 = quad, 2 = triangles, 3 = hybrid
forest = build_forest(comm, n_dims, refinement_level, 1)

max_number_faces = 4

number_trees = t8_forest_get_num_local_trees(forest)
println("rank $(MPI.Comm_rank(comm)): #trees $number_trees, #elements $(t8_forest_get_local_num_elements(forest)), #ghost_elements $(t8_forest_get_num_ghosts(forest))")

number_elements_global = t8_forest_get_global_num_elements(forest)
if MPI.Comm_rank(comm) == 0
    println("#global elements $number_elements_global")
end

mesh = T8codeMesh{n_dims}(forest, max_number_faces)

####################################################

advection_velocity = (0.2, -0.7)
equations = LinearScalarAdvectionEquation2D(advection_velocity)

function initial_condition_test(x, t, equations::LinearScalarAdvectionEquation2D)
    x1 = x[1]
    x2 = x[2]

    sigma = 0.05
    x1c = 0.5
    x2c = 0.5
    return exp(-0.5*((x1-x1c)^2 + (x2-x2c)^2)/sigma)
end
initial_condition = initial_condition_test

solver = FVMuscl(surface_flux=flux_lax_friedrichs)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)
if MPI.Comm_rank(comm) == 0
    @info "" semi
    show(semi.cache.elements[42])
end

fv_method_first_order(semi)

# Output the data to vtu files.
# prefix_forest_with_data = "hybrid_mesh_example"
# Trixi.output_data_to_vtu(mesh, semi, prefix_forest_with_data)

# ode = semidiscretize(semi, (0.0, 1.0));

# summary_callback = SummaryCallback()

# analysis_callback = AnalysisCallback(semi, interval=100)

# save_solution = SaveSolutionCallback(interval=100,
#                                      solution_variables=cons2prim)

# stepsize_callback = StepsizeCallback(cfl=1.6)

# callbacks = CallbackSet()#summary_callback, analysis_callback, save_solution, stepsize_callback)


###############################################################################
# run the simulation

# sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false),
#             dt=1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
#             save_everystep=false, callback=callbacks);
# summary_callback()

#
# Clean-up
#

t8_forest_unref(Ref(forest))
T8code.Libt8.sc_finalize()
# MPI.Finalize()
