# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin


mutable struct P4estMPIInterfaceContainer{NDIMS, uEltype<:Real, NDIMSP2} <: AbstractContainer
  u                ::Array{uEltype, NDIMSP2}       # [primary/secondary, variable, i, j, interface]
  local_element_ids::Vector{Int}                   # [interface]
  node_indices     ::Vector{NTuple{NDIMS, Symbol}} # [interface]
  local_sides      ::Vector{Int}                   # [interface]

  # internal `resize!`able storage
  _u           ::Vector{uEltype}
end

@inline nmpiinterfaces(interfaces::P4estMPIInterfaceContainer) = length(interfaces.local_sides)
@inline Base.ndims(::P4estMPIInterfaceContainer{NDIMS}) where NDIMS = NDIMS

function Base.resize!(mpi_interfaces::P4estMPIInterfaceContainer, capacity)
  @unpack _u, local_element_ids, node_indices, local_sides = mpi_interfaces

  n_dims = ndims(mpi_interfaces)
  n_nodes = size(interfaces.u, 3)
  n_variables = size(interfaces.u, 2)

  resize!(_u, 2 * n_variables * n_nodes^(n_dims-1) * capacity)
  interfaces.u = unsafe_wrap(Array, pointer(_u),
    (2, n_variables, ntuple(_ -> n_nodes, n_dims-1)..., capacity))

  resize!(local_element_ids, capacity)

  resize!(node_indices, capacity)

  resize!(local_sides, capacity)

  return nothing
end


# Create MPI interface container and initialize interface data
function init_mpi_interfaces(mesh::ParallelP4estMesh, equations, basis, elements)
  NDIMS = ndims(elements)
  uEltype = eltype(elements)

  # Initialize container
  n_mpi_interfaces = count_required_surfaces(mesh).mpi_interfaces

  _u = Vector{uEltype}(undef, 2 * nvariables(equations) * nnodes(basis)^(NDIMS-1) * n_mpi_interfaces)
  u = unsafe_wrap(Array, pointer(_u),
    (2, nvariables(equations), ntuple(_ -> nnodes(basis), NDIMS-1)..., n_mpi_interfaces))

  local_element_ids = Vector{Int}(undef, n_mpi_interfaces)

  node_indices = Vector{NTuple{NDIMS, Symbol}}(undef, n_mpi_interfaces)

  local_sides = Vector{Int}(undef, n_mpi_interfaces)

  mpi_interfaces = P4estMPIInterfaceContainer{NDIMS, uEltype, NDIMS+2}(
    u, local_element_ids, node_indices, local_sides, _u)

  init_mpi_interfaces!(mpi_interfaces, mesh)

  return mpi_interfaces
end

function init_mpi_interfaces!(mpi_interfaces, mesh::ParallelP4estMesh)
  init_surfaces!(nothing, nothing, nothing, mpi_interfaces, mesh)

  return mpi_interfaces
end

# Overload init! function for boundaries and regular interfaces since it must call the appropriate
# init_surfaces! function for parallel p4est meshes
function init_interfaces!(interfaces, mesh::ParallelP4estMesh)
  init_surfaces!(interfaces, nothing, nothing, nothing, mesh)

  return interfaces
end

function init_boundaries!(boundaries, mesh::ParallelP4estMesh)
  init_surfaces!(nothing, nothing, boundaries, nothing, mesh)

  return boundaries
end


function reinitialize_containers!(mesh::ParallelP4estMesh, equations, dg::DGSEM, cache)
  # Re-initialize elements container
  @unpack elements = cache
  resize!(elements, ncells(mesh))
  init_elements!(elements, mesh, dg.basis)

  required = count_required_surfaces(mesh)

  # resize interfaces container
  @unpack interfaces = cache
  resize!(interfaces, required.interfaces)

  # resize boundaries container
  @unpack boundaries = cache
  resize!(boundaries, required.boundaries)

  # resize mortars container
  @unpack mortars = cache
  resize!(mortars, required.mortars)

  # resize mpi_interfaces container
  @unpack mpi_interfaces = cache
  resize!(mpi_interfaces, required.mpi_interfaces)

  # re-initialize containers together to reduce
  # the number of iterations over the mesh in p4est
  init_surfaces!(interfaces, mortars, boundaries, mpi_interfaces, mesh)
end


# A helper struct used in initialization methods below
mutable struct ParallelInitSurfacesIterFaceUserData{Interfaces, Mortars, Boundaries, MPIInterfaces, Mesh}
  interfaces      ::Interfaces
  interface_id    ::Int
  mortars         ::Mortars
  mortar_id       ::Int
  boundaries      ::Boundaries
  boundary_id     ::Int
  mpi_interfaces  ::MPIInterfaces
  mpi_interface_id::Int
  mesh            ::Mesh
end

function ParallelInitSurfacesIterFaceUserData(interfaces, mortars, boundaries, mpi_interfaces, mesh)
  return ParallelInitSurfacesIterFaceUserData{
    typeof(interfaces), typeof(mortars), typeof(boundaries), typeof(mpi_interfaces), typeof(mesh)}(
      interfaces, 1, mortars, 1, boundaries, 1, mpi_interfaces, 1, mesh)
end


function init_surfaces_iter_face_parallel(info, user_data)
  # Unpack user_data
  data = unsafe_pointer_to_objref(Ptr{ParallelInitSurfacesIterFaceUserData}(user_data))

  # Function barrier because the unpacked user_data above is type-unstable
  init_surfaces_iter_face_inner(info, data)
end

# 2D
cfunction(::typeof(init_surfaces_iter_face_parallel), ::Val{2}) = @cfunction(init_surfaces_iter_face_parallel, Cvoid, (Ptr{p4est_iter_face_info_t}, Ptr{Cvoid}))
# 3D
cfunction(::typeof(init_surfaces_iter_face_parallel), ::Val{3}) = @cfunction(init_surfaces_iter_face_parallel, Cvoid, (Ptr{p8est_iter_face_info_t}, Ptr{Cvoid}))

# Function barrier for type stability, overload for parallel P4estMesh
function init_surfaces_iter_face_inner(info, user_data::ParallelInitSurfacesIterFaceUserData)
  @unpack interfaces, boundaries, mpi_interfaces = user_data

  if info.sides.elem_count == 2
    # Two neighboring elements => Interface or mortar

    # Extract surface data
    sides = (unsafe_load_side(info, 1), unsafe_load_side(info, 2))

    if sides[1].is_hanging == 0 && sides[2].is_hanging == 0
      # No hanging nodes => normal interface or MPI interface
      if sides[1].is.full.is_ghost == 1 || sides[2].is.full.is_ghost == 1 # remote side => MPI interface
        if mpi_interfaces !== nothing
          init_mpi_interfaces_iter_face_inner(info, sides, user_data)
        end
      else
        if interfaces !== nothing
          init_interfaces_iter_face_inner(info, sides, user_data)
        end
      end
    else
      # Hanging nodes => mortar
      error("ParallelP4estMesh does not support non-conforming meshes.")
    end
  elseif info.sides.elem_count == 1
    # One neighboring elements => boundary
    if boundaries !== nothing
      init_boundaries_iter_face_inner(info, user_data)
    end
  end

  return nothing
end

function init_surfaces!(interfaces, mortars, boundaries, mpi_interfaces, mesh::ParallelP4estMesh)
  # Let p4est iterate over all interfaces and call init_surfaces_iter_face
  iter_face_c = cfunction(init_surfaces_iter_face_parallel, Val(ndims(mesh)))
  user_data = ParallelInitSurfacesIterFaceUserData(interfaces, mortars, boundaries, mpi_interfaces,
                                                   mesh)

  iterate_p4est(mesh.p4est, user_data; iter_face_c=iter_face_c)

  return nothing
end


# Initialization of MPI interfaces after the function barrier
function init_mpi_interfaces_iter_face_inner(info, sides, user_data)
  @unpack mpi_interfaces, mpi_interface_id, mesh = user_data
  user_data.mpi_interface_id += 1

  if sides[1].is.full.is_ghost == 1
    local_side = 2
  elseif sides[2].is.full.is_ghost == 1
    local_side = 1
  else
    error("should not happen")
  end

  # Get local tree, one-based indexing
  tree = unsafe_load_tree(mesh.p4est, sides[local_side].treeid + 1)
  # Quadrant numbering offset of the local quadrant at this interface
  offset = tree.quadrants_offset
  tree_quad_id = sides[local_side].is.full.quadid # quadid in the local tree
  # ID of the local neighboring quad, cumulative over local trees
  local_quad_id = offset + tree_quad_id

  # p4est uses zero-based indexing, convert to one-based indexing
  mpi_interfaces.local_element_ids[mpi_interface_id] = local_quad_id + 1
  mpi_interfaces.local_sides[mpi_interface_id] = local_side

  # Face at which the interface lies
  faces = (sides[1].face, sides[2].face)

  # Save mpi_interfaces.node_indices dimension specific in containers_[23]d_parallel.jl
  init_mpi_interface_node_indices!(mpi_interfaces, faces, local_side, info.orientation,
                                   mpi_interface_id)

  return nothing
end


# Iterate over all interfaces and count
# - (inner) interfaces
# - mortars
# - boundaries
# - (MPI) interfaces at subdomain boundaries
# and collect the numbers in `user_data` in this order.
function count_surfaces_iter_face_parallel(info, user_data)
  if info.sides.elem_count == 2
    # Two neighboring elements => Interface or mortar

    # Extract surface data
    sides = (unsafe_load_side(info, 1), unsafe_load_side(info, 2))

    if sides[1].is_hanging == 0 && sides[2].is_hanging == 0
      # No hanging nodes => normal interface or MPI interface
      if sides[1].is.full.is_ghost == 1 || sides[2].is.full.is_ghost == 1 # remote side => MPI interface
        # Unpack user_data = [mpi_interface_count] and increment mpi_interface_count
        ptr = Ptr{Int}(user_data)
        id = unsafe_load(ptr, 4)
        unsafe_store!(ptr, id + 1, 4)
      else
        # Unpack user_data = [interface_count] and increment interface_count
        ptr = Ptr{Int}(user_data)
        id = unsafe_load(ptr, 1)
        unsafe_store!(ptr, id + 1, 1)
      end
    else
      # Hanging nodes => mortar
      error("ParallelP4estMesh does not support non-conforming meshes.")
    end
  elseif info.sides.elem_count == 1
    # One neighboring elements => boundary

    # Unpack user_data = [boundary_count] and increment boundary_count
    ptr = Ptr{Int}(user_data)
    id = unsafe_load(ptr, 3)
    unsafe_store!(ptr, id + 1, 3)
  end

  return nothing
end

# 2D
cfunction(::typeof(count_surfaces_iter_face_parallel), ::Val{2}) = @cfunction(count_surfaces_iter_face_parallel, Cvoid, (Ptr{p4est_iter_face_info_t}, Ptr{Cvoid}))
# 3D
cfunction(::typeof(count_surfaces_iter_face_parallel), ::Val{3}) = @cfunction(count_surfaces_iter_face_parallel, Cvoid, (Ptr{p8est_iter_face_info_t}, Ptr{Cvoid}))

function count_required_surfaces(mesh::ParallelP4estMesh)
  # Let p4est iterate over all interfaces and call count_surfaces_iter_face_parallel
  iter_face_c = cfunction(count_surfaces_iter_face_parallel, Val(ndims(mesh)))

  # interfaces, mortars, boundaries, mpi_interfaces
  user_data = [0, 0, 0, 0]

  iterate_p4est(mesh.p4est, user_data; iter_face_c=iter_face_c)

  # Return counters
  return (interfaces     = user_data[1],
          mortars        = user_data[2],
          boundaries     = user_data[3],
          mpi_interfaces = user_data[4])
end


end # @muladd