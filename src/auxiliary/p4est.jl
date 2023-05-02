# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin


"""
    init_p4est()

Initialize `p4est` by calling `p4est_init` and setting the log level to `SC_LP_ERROR`.
This function will check if `p4est` is already initialized
and if yes, do nothing, thus it is safe to call it multiple times.
"""
function init_p4est()
  p4est_package_id = P4est.package_id()
  if p4est_package_id >= 0
    return nothing
  end

  # Initialize `p4est` with log level ERROR to prevent a lot of output in AMR simulations
  p4est_init(C_NULL, SC_LP_ERROR)

  return nothing
end


# Convert sc_array of type T to Julia array
function unsafe_wrap_sc(::Type{T}, sc_array::Ptr{sc_array}) where T
  sc_array_obj = unsafe_load(sc_array)
  return unsafe_wrap_sc(T, sc_array_obj)
end

function unsafe_wrap_sc(::Type{T}, sc_array_obj::sc_array) where T
  elem_count = sc_array_obj.elem_count
  array = sc_array_obj.array

  return unsafe_wrap(Array, Ptr{T}(array), elem_count)
end


# Load the ith element (1-indexed) of an sc array of type T
function unsafe_load_sc(::Type{T}, sc_array::Ptr{sc_array}, i=1) where T
  sc_array_obj = unsafe_load(sc_array)
  return unsafe_load_sc(T, sc_array_obj, i)
end

function unsafe_load_sc(::Type{T}, sc_array_obj::sc_array, i=1) where T
  element_size = sc_array_obj.elem_size
  @assert element_size == sizeof(T)

  return unsafe_load(Ptr{T}(sc_array_obj.array), i)
end


# Create new `p4est` from a p4est_connectivity
# 2D
function new_p4est(connectivity::Ptr{p4est_connectivity_t}, initial_refinement_level)
  comm = P4est.uses_mpi() ? mpi_comm() : 0 # Use Trixi.jl's MPI communicator if p4est supports MPI
  p4est_new_ext(comm,
                connectivity,
                0, # No minimum initial qudrants per processor
                initial_refinement_level,
                true, # Refine uniformly
                2 * sizeof(Int), # Use Int-Vector of size 2 as quadrant user data
                C_NULL, # No init function
                C_NULL) # No user pointer
end

# 3D
function new_p4est(connectivity::Ptr{p8est_connectivity_t}, initial_refinement_level)
  comm = P4est.uses_mpi() ? mpi_comm() : 0 # Use Trixi.jl's MPI communicator if p4est supports MPI
  p8est_new_ext(comm, connectivity, 0, initial_refinement_level, true, 2 * sizeof(Int), C_NULL, C_NULL)
end


# Save `p4est` data to file
# 2D
function save_p4est!(file, p4est::Ptr{p4est_t})
  # Don't save user data of the quads
  p4est_save(file, p4est, false)
end

# 3D
function save_p4est!(file, p8est::Ptr{p8est_t})
  # Don't save user data of the quads
  p8est_save(file, p8est, false)
end


# Load `p4est` from file
# 2D
function load_p4est(file, ::Val{2})
  conn_vec = Vector{Ptr{p4est_connectivity_t}}(undef, 1)
  comm = P4est.uses_mpi() ? mpi_comm() : C_NULL # Use Trixi.jl's MPI communicator if p4est supports MPI
  p4est_load_ext(file, comm, 0, 0, 1, 0, C_NULL, pointer(conn_vec))
end

# 3D
function load_p4est(file, ::Val{3})
  conn_vec = Vector{Ptr{p8est_connectivity_t}}(undef, 1)
  comm = P4est.uses_mpi() ? mpi_comm() : C_NULL # Use Trixi.jl's MPI communicator if p4est supports MPI
  p8est_load_ext(file, comm, 0, 0, 1, 0, C_NULL, pointer(conn_vec))
end


# Read `p4est` connectivity from Abaqus mesh file (.inp)
# 2D
read_inp_p4est(meshfile, ::Val{2}) = p4est_connectivity_read_inp(meshfile)
# 3D
read_inp_p4est(meshfile, ::Val{3}) = p8est_connectivity_read_inp(meshfile)


# Refine `p4est` if refine_fn_c returns 1
# 2D
refine_p4est!(p4est::Ptr{p4est_t}, recursive, refine_fn_c, init_fn_c) = p4est_refine(p4est, recursive, refine_fn_c, init_fn_c)
# 3D
refine_p4est!(p8est::Ptr{p8est_t}, recursive, refine_fn_c, init_fn_c) = p8est_refine(p8est, recursive, refine_fn_c, init_fn_c)


# Refine `p4est` if coarsen_fn_c returns 1
# 2D
coarsen_p4est!(p4est::Ptr{p4est_t}, recursive, coarsen_fn_c, init_fn_c) = p4est_coarsen(p4est, recursive, coarsen_fn_c, init_fn_c)
# 3D
coarsen_p4est!(p8est::Ptr{p8est_t}, recursive, coarsen_fn_c, init_fn_c) = p8est_coarsen(p8est, recursive, coarsen_fn_c, init_fn_c)


# Create new ghost layer from p4est, only connections via faces are relevant
# 2D
ghost_new_p4est(p4est::Ptr{p4est_t}) = p4est_ghost_new(p4est, P4est.P4EST_CONNECT_FACE)
# 3D
# In 3D it is not sufficient to use `P8EST_CONNECT_FACE`. Consider the neighbor elements of a mortar
# in 3D. We have to determine which MPI ranks are involved in this mortar.
# ┌─────────────┬─────────────┐  ┌───────────────────────────┐
# │             │             │  │                           │
# │    small    │    small    │  │                           │
# │      3      │      4      │  │                           │
# │             │             │  │           large           │
# ├─────────────┼─────────────┤  │             5             │
# │             │             │  │                           │
# │    small    │    small    │  │                           │
# │      1      │      2      │  │                           │
# │             │             │  │                           │
# └─────────────┴─────────────┘  └───────────────────────────┘
# Suppose one process only owns element 1. Since element 4 is not connected to element 1 via a face,
# there is no guarantee that element 4 will be in the ghost layer, if it is constructed with
# `P8EST_CONNECT_FACE`. But if it is not in the ghost layer, it will not be available in
# `iterate_p4est` and thus we cannot determine its MPI rank
# (see https://github.com/cburstedde/p4est/blob/439bc9aae849555256ddfe4b03d1f9fe8d18ff0e/src/p8est_iterate.h#L66-L72).
ghost_new_p4est(p8est::Ptr{p8est_t}) = p8est_ghost_new(p8est, P4est.P8EST_CONNECT_FULL)

# Check if ghost layer is valid
# 2D
function ghost_is_valid_p4est(p4est::Ptr{p4est_t}, ghost_layer::Ptr{p4est_ghost_t})
  return p4est_ghost_is_valid(p4est, ghost_layer)
end
# 3D
function ghost_is_valid_p4est(p4est::Ptr{p8est_t}, ghost_layer::Ptr{p8est_ghost_t})
  return p8est_ghost_is_valid(p4est, ghost_layer)
end

# Destroy ghost layer
# 2D
ghost_destroy_p4est(ghost_layer::Ptr{p4est_ghost_t}) = p4est_ghost_destroy(ghost_layer)
# 3D
ghost_destroy_p4est(ghost_layer::Ptr{p8est_ghost_t}) = p8est_ghost_destroy(ghost_layer)


# Let `p4est` iterate over each cell volume and cell face.
# Call iter_volume_c for each cell and iter_face_c for each face.
# 2D
function iterate_p4est(p4est::Ptr{p4est_t}, user_data; ghost_layer=C_NULL,
                       iter_volume_c=C_NULL, iter_face_c=C_NULL)
  if user_data === C_NULL
    user_data_ptr = user_data
  elseif user_data isa AbstractArray
    user_data_ptr = pointer(user_data)
  else
    user_data_ptr = pointer_from_objref(user_data)
  end

  GC.@preserve user_data begin
    p4est_iterate(p4est,
                  ghost_layer,
                  user_data_ptr,
                  iter_volume_c, # iter_volume
                  iter_face_c, # iter_face
                  C_NULL) # iter_corner
  end

  return nothing
end

# 3D
function iterate_p4est(p8est::Ptr{p8est_t}, user_data; ghost_layer=C_NULL,
                       iter_volume_c=C_NULL, iter_face_c=C_NULL)
  if user_data === C_NULL
    user_data_ptr = user_data
  elseif user_data isa AbstractArray
    user_data_ptr = pointer(user_data)
  else
    user_data_ptr = pointer_from_objref(user_data)
  end

  GC.@preserve user_data begin
    p8est_iterate(p8est,
                  ghost_layer,
                  user_data_ptr,
                  iter_volume_c, # iter_volume
                  iter_face_c, # iter_face
                  C_NULL, # iter_edge
                  C_NULL) # iter_corner
  end

  return nothing
end


# Load i-th element of the sc_array info.sides of the type p[48]est_iter_face_side_t
# 2D version
function unsafe_load_side(info::Ptr{p4est_iter_face_info_t}, i=1)
  return unsafe_load_sc(p4est_iter_face_side_t, unsafe_load(info).sides, i)
end

# 3D version
function unsafe_load_side(info::Ptr{p8est_iter_face_info_t}, i=1)
  return unsafe_load_sc(p8est_iter_face_side_t, unsafe_load(info).sides, i)
end


# Load i-th element of the sc_array p4est.trees of the type p[48]est_tree_t
# 2D version
function unsafe_load_tree(p4est::Ptr{p4est_t}, i=1)
  return unsafe_load_sc(p4est_tree_t, unsafe_load(p4est).trees, i)
end

# 3D version
function unsafe_load_tree(p8est::Ptr{p8est_t}, i=1)
  return unsafe_load_sc(p8est_tree_t, unsafe_load(p8est).trees, i)
end


end # @muladd
