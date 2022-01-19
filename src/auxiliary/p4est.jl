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
  if p4est_package_id()[] >= 0
    return nothing
  end

  # Initialize `p4est` with log level ERROR to prevent a lot of output in AMR simulations
  p4est_init(C_NULL, SC_LP_ERROR)

  return nothing
end


# Convert sc_array of type T to Julia array
function unsafe_wrap_sc(::Type{T}, sc_array) where T
  element_count = sc_array.elem_count

  return [unsafe_load_sc(T, sc_array, i) for i in 1:element_count]
end


# Load the ith element (1-indexed) of an sc array of type T
function unsafe_load_sc(::Type{T}, sc_array, i=1) where T
  element_size = sc_array.elem_size

  @assert element_size == sizeof(T)

  return unsafe_load(Ptr{T}(sc_array.array), i)
end


# Create new `p4est` from a p4est_connectivity
# 2D
function new_p4est(conn::Ptr{p4est_connectivity_t}, initial_refinement_level)
  p4est_new_ext(0, # No MPI communicator
                conn,
                0, # No minimum initial qudrants per processor
                initial_refinement_level,
                true, # Refine uniformly
                2 * sizeof(Int), # Use Int-Vector of size 2 as quadrant user data
                C_NULL, # No init function
                C_NULL) # No user pointer
end

# 3D
function new_p4est(conn::Ptr{p8est_connectivity_t}, initial_refinement_level)
  p8est_new_ext(0, conn, 0, initial_refinement_level, true, 2 * sizeof(Int), C_NULL, C_NULL)
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
  p4est_load(file, C_NULL, 0, false, C_NULL, pointer(conn_vec))
end

# 3D
function load_p4est(file, ::Val{3})
  conn_vec = Vector{Ptr{p8est_connectivity_t}}(undef, 1)
  p8est_load(file, C_NULL, 0, false, C_NULL, pointer(conn_vec))
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


# Let `p4est` iterate over each cell volume and cell face.
# Call iter_volume_c for each cell and iter_face_c for each face.
# 2D
function iterate_p4est(p4est::Ptr{p4est_t}, user_data;
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
                  C_NULL, # ghost layer
                  user_data_ptr,
                  iter_volume_c, # iter_volume
                  iter_face_c, # iter_face
                  C_NULL) # iter_corner
  end

  return nothing
end

# 3D
function iterate_p4est(p8est::Ptr{p8est_t}, user_data;
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
                  C_NULL, # ghost layer
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
  return unsafe_load_sc(p4est_iter_face_side_t, info.sides, i)
end

# 3D version
function unsafe_load_side(info::Ptr{p8est_iter_face_info_t}, i=1)
  return unsafe_load_sc(p8est_iter_face_side_t, info.sides, i)
end


# Load i-th element of the sc_array p4est.trees of the type p[48]est_tree_t
# 2D version
function unsafe_load_tree(p4est::Ptr{p4est_t}, i=1)
  return unsafe_load_sc(p4est_tree_t, p4est.trees, i)
end

# 3D version
function unsafe_load_tree(p8est::Ptr{p8est_t}, i=1)
  return unsafe_load_sc(p8est_tree_t, p8est.trees, i)
end


end # @muladd
