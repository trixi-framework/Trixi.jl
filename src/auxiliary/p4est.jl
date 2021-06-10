"""
    init_p4est()

Initialize p4est by calling `p4est_init` and setting the log level to `SC_LP_ERROR`.
This function will check if p4est is already initialized
and if yes, do nothing, thus it is safe to call it multiple times.
"""
function init_p4est()
  if p4est_package_id()[] >= 0
    return nothing
  end

  # Initialize p4est with log level ERROR to prevent a lot of output in AMR simulations
  p4est_init(C_NULL, SC_LP_ERROR)

  return nothing
end


function register_destroy_p4est_structs(p4est::Ptr{p4est_t}, conn)
  # Destroy p4est structs at exit of Julia
  function destroy_p4est_structs()
    p4est_destroy(p4est)
    p4est_connectivity_destroy(conn)
  end

  atexit(destroy_p4est_structs)
end

function register_destroy_p4est_structs(p8est::Ptr{p8est_t}, conn)
  # Destroy p4est structs at exit of Julia
  function destroy_p8est_structs()
    p8est_destroy(p8est)
    p8est_connectivity_destroy(conn)
  end

  atexit(destroy_p8est_structs)
end


# Convert sc_array of type T to Julia array
function unsafe_wrap_sc(::Type{T}, sc_array) where T
  element_count = sc_array.elem_count
  element_size = sc_array.elem_size

  @assert element_size == sizeof(T)

  return [unsafe_wrap(T, sc_array.array + element_size * i) for i in 0:element_count-1]
end


# Load the ith element (1-indexed) of an sc array of type T
function unsafe_load_sc(::Type{T}, sc_array, i=1) where T
  element_size = sc_array.elem_size

  @assert element_size == sizeof(T)

  return unsafe_wrap(T, sc_array.array + element_size * (i - 1))
end


# Let p4est iterate over all interfaces and execute the C function iter_face_c
function iterate_faces(p4est::Ptr{p4est_t}, iter_face_c, user_data)
  GC.@preserve user_data begin
    p4est_iterate(p4est,
                  C_NULL, # ghost layer
                  pointer(user_data),
                  C_NULL, # iter_volume
                  iter_face_c, # iter_face
                  C_NULL) # iter_corner
  end

  return nothing
end

function iterate_faces(p8est::Ptr{p8est_t}, iter_face_c, user_data)
  GC.@preserve user_data begin
    p8est_iterate(p8est,
                  C_NULL, # ghost layer
                  pointer(user_data),
                  C_NULL, # iter_volume
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
