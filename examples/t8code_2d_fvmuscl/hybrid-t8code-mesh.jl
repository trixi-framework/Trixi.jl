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
using T8code.Libt8: sc_init
using T8code.Libt8: sc_free
using T8code.Libt8: sc_finalize
using T8code.Libt8: sc_array_new_data
using T8code.Libt8: sc_array_destroy
using T8code.Libt8: SC_LP_ESSENTIAL
using T8code.Libt8: SC_LP_PRODUCTION

const NDIMS = 2
const MAX_NUM_FACES = 4

# The uniform refinement level of the forest.
const refinement_level = 4

# The prefix for our output files.
const prefix_forest_with_data = "hybrid_mesh_example"

# The data that we want to store for each element.
struct data_per_element_t
  level             :: Cint
  volume            :: Cdouble
  midpoint          :: NTuple{NDIMS,Cdouble}
  dx                :: Cdouble # Characteristic length (for CFL condition).

  num_faces         :: Cint
  face_areas        :: NTuple{MAX_NUM_FACES,Cdouble}
  face_normals      :: NTuple{MAX_NUM_FACES*NDIMS,Cdouble}
  face_connectivity :: NTuple{MAX_NUM_FACES,t8_locidx_t} # ids of the face neighbors

  # State variables.
  scalar            :: Cdouble
end

function pretty_print(data :: data_per_element_t)

  println("level              = ", data.level)
  println("volume             = ", data.volume)
  println("midpoint           = ", data.midpoint)
  println("dx                 = ", data.dx)
  println("num_faces          = ", data.num_faces)
  println("face_areas         = ", data.face_areas)
  println("face_normals       = ", data.face_normals)
  println("face_connectivity  = ", data.face_connectivity)
  println("scalar             = ", data.scalar)

end

function initial_condition(coords)
  x = coords[1]
  y = coords[2]

  sigma = 0.05
  xc = 0.5
  yc = 0.5
  return exp(-0.5*((x-xc)^2 + (y-yc)^2)/sigma)
end

# Directly ported from: `src/t8_cmesh/t8_cmesh_examples.c: t8_cmesh_new_periodic_hybrid`.
function cmesh_new_periodic_hybrid(comm) :: t8_cmesh_t
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
  linear_geom = t8_geometry_linear_new(NDIMS)

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

function build_forest(comm, level)

  # Periodic mesh of quads.
  # cmesh = t8_cmesh_new_periodic(comm, NDIMS)

  # Periodic mesh of triangles.
  # cmesh = t8_cmesh_new_periodic_tri(comm)

  # Periodic mesh of quads and triangles.
  # cmesh = t8_cmesh_new_periodic_hybrid(comm) # The `t8code` version does exactly the same.
  cmesh = cmesh_new_periodic_hybrid(comm)

  scheme = t8_scheme_new_default_cxx()

  let do_face_ghost = 1
    forest = t8_forest_new_uniform(cmesh, scheme, level, do_face_ghost, comm)
    return forest
  end
end

# Allocate and fill the element data array. Returns a pointer to the array
# which is then ownded by the calling scope.
function create_element_data(forest)
  # Check that the forest is a committed.
  @assert(t8_forest_is_committed(forest) == 1)

  # Get the number of local elements of forest.
  num_local_elements = t8_forest_get_local_num_elements(forest)
  # Get the number of ghost elements of forest.
  num_ghost_elements = t8_forest_get_num_ghosts(forest)

  # Build an array of our data that is as long as the number of elements plus
  # the number of ghosts.
  element_data = Array{data_per_element_t}(undef, num_local_elements + num_ghost_elements)

  # Get the number of trees that have elements of this process.
  num_local_trees = t8_forest_get_num_local_trees(forest)

  midpoint = Vector{Cdouble}(undef,NDIMS)

  face_areas = Vector{Cdouble}(undef,MAX_NUM_FACES)
  face_normals = Matrix{Cdouble}(undef,3,MAX_NUM_FACES) # Need NDIMS=3 for t8code API. Also, consider that Julia is column major.
  face_connectivity = Vector{t8_locidx_t}(undef,MAX_NUM_FACES)

  # Loop over all local trees in the forest.
  current_index = 0
  for itree = 0:num_local_trees-1
    tree_class = t8_forest_get_tree_class(forest, itree)
    eclass_scheme = t8_forest_get_eclass_scheme(forest, tree_class)

    # Get the number of elements of this tree.
    num_elements_in_tree = t8_forest_get_tree_num_elements(forest, itree)

    # Loop over all local elements in the tree.
    for ielement = 0:num_elements_in_tree-1
      current_index += 1 # Note: Julia has 1-based indexing, while C/C++ starts with 0.

      element = t8_forest_get_element_in_tree(forest, itree, ielement)

      level = t8_element_level(eclass_scheme, element)
      volume = t8_forest_element_volume(forest, itree, element)

      t8_forest_element_centroid(forest, itree, element, pointer(midpoint))

      # Characteristic length of the element. It is an approximation since only
      # for the element type `lines` this can be exact.
      dx = t8_forest_element_diam(forest, itree, element)

      # Loop over all faces of an element.
      num_faces = t8_element_num_faces(eclass_scheme, element)

      # Set default value.
      face_connectivity .= -1

      for iface = 1:num_faces
        face_areas[iface] = t8_forest_element_face_area(forest, itree, element, iface-1) # C++ is zero-indexed
        t8_forest_element_face_normal(forest, itree, element, iface-1, @views(face_normals[:,iface]))

        # [ugly API, needs rework :/]
        neighids_ref = Ref{Ptr{t8_locidx_t}}()
        neighbors_ref = Ref{Ptr{Ptr{t8_element}}}()
        neigh_scheme_ref = Ref{Ptr{t8_eclass_scheme}}()

        dual_faces_ref = Ref{Ptr{Cint}}()
        num_neighbors_ref = Ref{Cint}()

        forest_is_balanced = Cint(1)

        t8_forest_leaf_face_neighbors(forest, itree, element,
          neighbors_ref, iface-1, dual_faces_ref, num_neighbors_ref,
          neighids_ref, neigh_scheme_ref, forest_is_balanced)

        num_neighbors = num_neighbors_ref[]
        dual_faces    = 1 .+ unsafe_wrap(Array, dual_faces_ref[], num_neighbors)
        neighids      = 1 .+ unsafe_wrap(Array, neighids_ref[], num_neighbors)
        neighbors     = unsafe_wrap(Array, neighbors_ref[], num_neighbors)
        neigh_scheme  = neigh_scheme_ref[]

        face_connectivity[iface] = neighids[1]

        # Free allocated memory.
        sc_free(t8_get_package_id(), neighbors_ref[])
        sc_free(t8_get_package_id(), dual_faces_ref[])
        sc_free(t8_get_package_id(), neighids_ref[])
        # [/ugly API]
      end

      # Some 'interesting' height function.
      scalar = initial_condition(midpoint)

      element_data[current_index] = data_per_element_t(
        level,
        volume,
        Tuple(midpoint),
        dx,
        num_faces,
        Tuple(face_areas),
        Tuple(@views(face_normals[1:2,:])),
        Tuple(face_connectivity),
        scalar
      )
    end
  end

  return element_data
end

# Each process has computed the data entries for its local elements.  In order
# to get the values for the ghost elements, we use
# t8_forest_ghost_exchange_data.  Calling this function will fill all the ghost
# entries of our element data array with the value on the process that owns the
# corresponding element. */
function exchange_ghost_data(forest, element_data)
  # t8_forest_ghost_exchange_data expects an sc_array (of length num_local_elements + num_ghosts).
  # We wrap our data array to an sc_array.
  sc_array_wrapper = sc_array_new_data(pointer(element_data), sizeof(data_per_element_t), length(element_data))

  # Carry out the data exchange. The entries with indices > num_local_elements will get overwritten.
  t8_forest_ghost_exchange_data(forest, sc_array_wrapper)

  # Destroy the wrapper array. This will not free the data memory since we used sc_array_new_data.
  sc_array_destroy(sc_array_wrapper)
end

# Write the forest as vtu and also write the element's volumes in the file.
#
# t8code supports writing element based data to vtu as long as its stored
# as doubles. Each of the data fields to write has to be provided in its own
# array of length num_local_elements.
# We support two types: T8_VTK_SCALAR - One double per element.
#                  and  T8_VTK_VECTOR - Three doubles per element.
function output_data_to_vtu(forest, element_data, prefix)
  num_elements = length(element_data)

  # We need to allocate a new array to store the data on their own.
  # These arrays have one entry per local element.
  scalars = Vector{Cdouble}(undef, num_elements)

  # Copy the elment's volumes from our data array to the output array.
  for ielem = 1:num_elements
    scalars[ielem] = element_data[ielem].scalar
  end

  vtk_data = [
    t8_vtk_data_field_t(
      T8_VTK_SCALAR,
      NTuple{8192, Cchar}(rpad("scalar\0", 8192, ' ')),
      pointer(scalars),
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
  t8_forest_write_vtk_ext(forest, prefix, write_treeid, write_mpirank,
                           write_level, write_element_id, write_ghosts,
                           0, 0, num_data, pointer(vtk_data))
end

#
# Initialization.
#

# Initialize MPI. This has to happen before we initialize sc or t8code.
mpiret = MPI.Init()

# We will use MPI_COMM_WORLD as a communicator.
comm = MPI.COMM_WORLD

# Initialize the sc library, has to happen before we initialize t8code.
sc_init(comm, 1, 1, C_NULL, SC_LP_ESSENTIAL)
# Initialize t8code with log level SC_LP_PRODUCTION. See sc.h for more info on the log levels.
t8_init(SC_LP_PRODUCTION)

# Initialize an adapted forest with periodic boundaries.
forest = build_forest(comm, refinement_level)

#
# Data handling and computation.
#

# Build data array and gather data for the local elements.
element_data = create_element_data(forest)

# Exchange the neighboring data at MPI process boundaries.
exchange_ghost_data(forest, element_data)

# Output the data to vtu files.
output_data_to_vtu(forest, element_data, prefix_forest_with_data)

if MPI.Comm_rank(comm) == 0
  println("")
  pretty_print(element_data[42])
  println("")
end

mesh = T8codeMesh{2}(forest, element_data)

#
# Clean-up
#

# t8_forest_unref(Ref(forest))
sc_finalize()