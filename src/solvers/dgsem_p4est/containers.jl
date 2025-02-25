# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

mutable struct P4estElementContainer{NDIMS, RealT <: Real, uEltype <: Real, NDIMSP1,
                                     NDIMSP2, NDIMSP3} <: AbstractContainer
    # Physical coordinates at each node
    node_coordinates::Array{RealT, NDIMSP2}   # [orientation, node_i, node_j, node_k, element]
    # Jacobian matrix of the transformation
    # [jacobian_i, jacobian_j, node_i, node_j, node_k, element] where jacobian_i is the first index of the Jacobian matrix,...
    jacobian_matrix::Array{RealT, NDIMSP3}
    # Contravariant vectors, scaled by J, in Kopriva's blue book called Ja^i_n (i index, n dimension)
    contravariant_vectors::Array{RealT, NDIMSP3}   # [dimension, index, node_i, node_j, node_k, element]
    # 1/J where J is the Jacobian determinant (determinant of Jacobian matrix)
    inverse_jacobian::Array{RealT, NDIMSP1}   # [node_i, node_j, node_k, element]
    # Buffer for calculated surface flux
    surface_flux_values::Array{uEltype, NDIMSP2} # [variable, i, j, direction, element]

    # internal `resize!`able storage
    _node_coordinates::Vector{RealT}
    _jacobian_matrix::Vector{RealT}
    _contravariant_vectors::Vector{RealT}
    _inverse_jacobian::Vector{RealT}
    _surface_flux_values::Vector{uEltype}
end

@inline function nelements(elements::P4estElementContainer)
    size(elements.node_coordinates, ndims(elements) + 2)
end
@inline Base.ndims(::P4estElementContainer{NDIMS}) where {NDIMS} = NDIMS
@inline function Base.eltype(::P4estElementContainer{NDIMS, RealT, uEltype}) where {
                                                                                    NDIMS,
                                                                                    RealT,
                                                                                    uEltype
                                                                                    }
    uEltype
end

# Only one-dimensional `Array`s are `resize!`able in Julia.
# Hence, we use `Vector`s as internal storage and `resize!`
# them whenever needed. Then, we reuse the same memory by
# `unsafe_wrap`ping multi-dimensional `Array`s around the
# internal storage.
function Base.resize!(elements::P4estElementContainer, capacity)
    @unpack _node_coordinates, _jacobian_matrix, _contravariant_vectors,
    _inverse_jacobian, _surface_flux_values = elements

    n_dims = ndims(elements)
    n_nodes = size(elements.node_coordinates, 2)
    n_variables = size(elements.surface_flux_values, 1)

    resize!(_node_coordinates, n_dims * n_nodes^n_dims * capacity)
    elements.node_coordinates = unsafe_wrap(Array, pointer(_node_coordinates),
                                            (n_dims, ntuple(_ -> n_nodes, n_dims)...,
                                             capacity))

    resize!(_jacobian_matrix, n_dims^2 * n_nodes^n_dims * capacity)
    elements.jacobian_matrix = unsafe_wrap(Array, pointer(_jacobian_matrix),
                                           (n_dims, n_dims,
                                            ntuple(_ -> n_nodes, n_dims)..., capacity))

    resize!(_contravariant_vectors, length(_jacobian_matrix))
    elements.contravariant_vectors = unsafe_wrap(Array, pointer(_contravariant_vectors),
                                                 size(elements.jacobian_matrix))

    resize!(_inverse_jacobian, n_nodes^n_dims * capacity)
    elements.inverse_jacobian = unsafe_wrap(Array, pointer(_inverse_jacobian),
                                            (ntuple(_ -> n_nodes, n_dims)..., capacity))

    resize!(_surface_flux_values,
            n_variables * n_nodes^(n_dims - 1) * (n_dims * 2) * capacity)
    elements.surface_flux_values = unsafe_wrap(Array, pointer(_surface_flux_values),
                                               (n_variables,
                                                ntuple(_ -> n_nodes, n_dims - 1)...,
                                                n_dims * 2, capacity))

    return nothing
end

# Create element container and initialize element data
function init_elements(mesh::Union{P4estMesh{NDIMS, NDIMS, RealT},
                                   T8codeMesh{NDIMS, RealT}},
                       equations,
                       basis,
                       ::Type{uEltype}) where {NDIMS, RealT <: Real, uEltype <: Real}
    nelements = ncells(mesh)

    _node_coordinates = Vector{RealT}(undef, NDIMS * nnodes(basis)^NDIMS * nelements)
    node_coordinates = unsafe_wrap(Array, pointer(_node_coordinates),
                                   (NDIMS, ntuple(_ -> nnodes(basis), NDIMS)...,
                                    nelements))

    _jacobian_matrix = Vector{RealT}(undef, NDIMS^2 * nnodes(basis)^NDIMS * nelements)
    jacobian_matrix = unsafe_wrap(Array, pointer(_jacobian_matrix),
                                  (NDIMS, NDIMS, ntuple(_ -> nnodes(basis), NDIMS)...,
                                   nelements))

    _contravariant_vectors = similar(_jacobian_matrix)
    contravariant_vectors = unsafe_wrap(Array, pointer(_contravariant_vectors),
                                        size(jacobian_matrix))

    _inverse_jacobian = Vector{RealT}(undef, nnodes(basis)^NDIMS * nelements)
    inverse_jacobian = unsafe_wrap(Array, pointer(_inverse_jacobian),
                                   (ntuple(_ -> nnodes(basis), NDIMS)..., nelements))

    _surface_flux_values = Vector{uEltype}(undef,
                                           nvariables(equations) *
                                           nnodes(basis)^(NDIMS - 1) * (NDIMS * 2) *
                                           nelements)
    surface_flux_values = unsafe_wrap(Array, pointer(_surface_flux_values),
                                      (nvariables(equations),
                                       ntuple(_ -> nnodes(basis), NDIMS - 1)...,
                                       NDIMS * 2, nelements))

    elements = P4estElementContainer{NDIMS, RealT, uEltype, NDIMS + 1, NDIMS + 2,
                                     NDIMS + 3}(node_coordinates, jacobian_matrix,
                                                contravariant_vectors,
                                                inverse_jacobian, surface_flux_values,
                                                _node_coordinates, _jacobian_matrix,
                                                _contravariant_vectors,
                                                _inverse_jacobian, _surface_flux_values)

    init_elements!(elements, mesh, basis)
    return elements
end

mutable struct P4estInterfaceContainer{NDIMS, uEltype <: Real, NDIMSP2} <:
               AbstractContainer
    u::Array{uEltype, NDIMSP2}       # [primary/secondary, variable, i, j, interface]
    neighbor_ids::Matrix{Int}                   # [primary/secondary, interface]
    node_indices::Matrix{NTuple{NDIMS, Symbol}} # [primary/secondary, interface]

    # internal `resize!`able storage
    _u::Vector{uEltype}
    _neighbor_ids::Vector{Int}
    _node_indices::Vector{NTuple{NDIMS, Symbol}}
end

@inline function ninterfaces(interfaces::P4estInterfaceContainer)
    size(interfaces.neighbor_ids, 2)
end
@inline Base.ndims(::P4estInterfaceContainer{NDIMS}) where {NDIMS} = NDIMS

# See explanation of Base.resize! for the element container
function Base.resize!(interfaces::P4estInterfaceContainer, capacity)
    @unpack _u, _neighbor_ids, _node_indices = interfaces

    n_dims = ndims(interfaces)
    n_nodes = size(interfaces.u, 3)
    n_variables = size(interfaces.u, 2)

    resize!(_u, 2 * n_variables * n_nodes^(n_dims - 1) * capacity)
    interfaces.u = unsafe_wrap(Array, pointer(_u),
                               (2, n_variables, ntuple(_ -> n_nodes, n_dims - 1)...,
                                capacity))

    resize!(_neighbor_ids, 2 * capacity)
    interfaces.neighbor_ids = unsafe_wrap(Array, pointer(_neighbor_ids), (2, capacity))

    resize!(_node_indices, 2 * capacity)
    interfaces.node_indices = unsafe_wrap(Array, pointer(_node_indices), (2, capacity))

    return nothing
end

# Create interface container and initialize interface data.
function init_interfaces(mesh::Union{P4estMesh, T8codeMesh}, equations, basis, elements)
    NDIMS = ndims(elements)
    uEltype = eltype(elements)

    # Initialize container
    n_interfaces = count_required_surfaces(mesh).interfaces

    _u = Vector{uEltype}(undef,
                         2 * nvariables(equations) * nnodes(basis)^(NDIMS - 1) *
                         n_interfaces)
    u = unsafe_wrap(Array, pointer(_u),
                    (2, nvariables(equations), ntuple(_ -> nnodes(basis), NDIMS - 1)...,
                     n_interfaces))

    _neighbor_ids = Vector{Int}(undef, 2 * n_interfaces)
    neighbor_ids = unsafe_wrap(Array, pointer(_neighbor_ids), (2, n_interfaces))

    _node_indices = Vector{NTuple{NDIMS, Symbol}}(undef, 2 * n_interfaces)
    node_indices = unsafe_wrap(Array, pointer(_node_indices), (2, n_interfaces))

    interfaces = P4estInterfaceContainer{NDIMS, uEltype, NDIMS + 2}(u, neighbor_ids,
                                                                    node_indices,
                                                                    _u, _neighbor_ids,
                                                                    _node_indices)

    init_interfaces!(interfaces, mesh)

    return interfaces
end

function init_interfaces!(interfaces, mesh::P4estMesh)
    init_surfaces!(interfaces, nothing, nothing, mesh)

    return interfaces
end

mutable struct P4estBoundaryContainer{NDIMS, uEltype <: Real, NDIMSP1} <:
               AbstractContainer
    u::Array{uEltype, NDIMSP1}       # [variables, i, j, boundary]
    neighbor_ids::Vector{Int}                   # [boundary]
    node_indices::Vector{NTuple{NDIMS, Symbol}} # [boundary]
    name::Vector{Symbol}                # [boundary]

    # internal `resize!`able storage
    _u::Vector{uEltype}
end

@inline function nboundaries(boundaries::P4estBoundaryContainer)
    length(boundaries.neighbor_ids)
end
@inline Base.ndims(::P4estBoundaryContainer{NDIMS}) where {NDIMS} = NDIMS

# See explanation of Base.resize! for the element container
function Base.resize!(boundaries::P4estBoundaryContainer, capacity)
    @unpack _u, neighbor_ids, node_indices, name = boundaries

    n_dims = ndims(boundaries)
    n_nodes = size(boundaries.u, 2)
    n_variables = size(boundaries.u, 1)

    resize!(_u, n_variables * n_nodes^(n_dims - 1) * capacity)
    boundaries.u = unsafe_wrap(Array, pointer(_u),
                               (n_variables, ntuple(_ -> n_nodes, n_dims - 1)...,
                                capacity))

    resize!(neighbor_ids, capacity)

    resize!(node_indices, capacity)

    resize!(name, capacity)

    return nothing
end

# Create interface container and initialize interface data in `elements`.
function init_boundaries(mesh::Union{P4estMesh, T8codeMesh}, equations, basis, elements)
    NDIMS = ndims(elements)
    uEltype = eltype(elements)

    # Initialize container
    n_boundaries = count_required_surfaces(mesh).boundaries

    _u = Vector{uEltype}(undef,
                         nvariables(equations) * nnodes(basis)^(NDIMS - 1) *
                         n_boundaries)
    u = unsafe_wrap(Array, pointer(_u),
                    (nvariables(equations), ntuple(_ -> nnodes(basis), NDIMS - 1)...,
                     n_boundaries))

    neighbor_ids = Vector{Int}(undef, n_boundaries)
    node_indices = Vector{NTuple{NDIMS, Symbol}}(undef, n_boundaries)
    names = Vector{Symbol}(undef, n_boundaries)

    boundaries = P4estBoundaryContainer{NDIMS, uEltype, NDIMS + 1}(u, neighbor_ids,
                                                                   node_indices, names,
                                                                   _u)

    if n_boundaries > 0
        init_boundaries!(boundaries, mesh)
    end

    return boundaries
end

function init_boundaries!(boundaries, mesh::P4estMesh)
    init_surfaces!(nothing, nothing, boundaries, mesh)

    return boundaries
end

# Function barrier for type stability
function init_boundaries_iter_face_inner(info_pw, boundaries, boundary_id, mesh)
    # Extract boundary data
    side_pw = load_pointerwrapper_side(info_pw)
    # Get local tree, one-based indexing
    tree_pw = load_pointerwrapper_tree(mesh.p4est, side_pw.treeid[] + 1)
    # Quadrant numbering offset of this quadrant
    offset = tree_pw.quadrants_offset[]

    # Verify before accessing is.full, but this should never happen
    @assert side_pw.is_hanging[] == false

    local_quad_id = side_pw.is.full.quadid[]
    # Global ID of this quad
    quad_id = offset + local_quad_id

    # Write data to boundaries container
    # `p4est` uses zero-based indexing; convert to one-based indexing
    boundaries.neighbor_ids[boundary_id] = quad_id + 1

    # Face at which the boundary lies
    face = side_pw.face[]

    # Save boundaries.node_indices dimension specific in containers_[23]d.jl
    init_boundary_node_indices!(boundaries, face, boundary_id)

    # One-based indexing
    boundaries.name[boundary_id] = mesh.boundary_names[face + 1, side_pw.treeid[] + 1]

    return nothing
end

# Container data structure (structure-of-arrays style) for DG L2 mortars
#
# The positions used in `neighbor_ids` are 1:3 (in 2D) or 1:5 (in 3D), where 1:2 (in 2D)
# or 1:4 (in 3D) are the small elements numbered in z-order and 3 or 5 is the large element.
# The solution values on the mortar element are saved in `u`, where `position` is the number
# of the small element that corresponds to the respective part of the mortar element.
# The first dimension `small/large side` takes 1 for small side and 2 for large side.
#
# Illustration of the positions in `neighbor_ids` in 3D, where ξ and η are the local coordinates
# of the mortar element, which are precisely the local coordinates that span
# the surface of the smaller side.
# Note that the orientation in the physical space is completely irrelevant here.
#   ┌─────────────┬─────────────┐  ┌───────────────────────────┐
#   │             │             │  │                           │
#   │    small    │    small    │  │                           │
#   │      3      │      4      │  │                           │
#   │             │             │  │           large           │
#   ├─────────────┼─────────────┤  │             5             │
# η │             │             │  │                           │
#   │    small    │    small    │  │                           │
# ↑ │      1      │      2      │  │                           │
# │ │             │             │  │                           │
# │ └─────────────┴─────────────┘  └───────────────────────────┘
# │
# ⋅────> ξ
mutable struct P4estMortarContainer{NDIMS, uEltype <: Real, NDIMSP1, NDIMSP3} <:
               AbstractContainer
    u::Array{uEltype, NDIMSP3} # [small/large side, variable, position, i, j, mortar]
    neighbor_ids::Matrix{Int}             # [position, mortar]
    node_indices::Matrix{NTuple{NDIMS, Symbol}} # [small/large, mortar]

    # internal `resize!`able storage
    _u::Vector{uEltype}
    _neighbor_ids::Vector{Int}
    _node_indices::Vector{NTuple{NDIMS, Symbol}}
end

@inline nmortars(mortars::P4estMortarContainer) = size(mortars.neighbor_ids, 2)
@inline Base.ndims(::P4estMortarContainer{NDIMS}) where {NDIMS} = NDIMS

# See explanation of Base.resize! for the element container
function Base.resize!(mortars::P4estMortarContainer, capacity)
    @unpack _u, _neighbor_ids, _node_indices = mortars

    n_dims = ndims(mortars)
    n_nodes = size(mortars.u, 4)
    n_variables = size(mortars.u, 2)

    resize!(_u, 2 * n_variables * 2^(n_dims - 1) * n_nodes^(n_dims - 1) * capacity)
    mortars.u = unsafe_wrap(Array, pointer(_u),
                            (2, n_variables, 2^(n_dims - 1),
                             ntuple(_ -> n_nodes, n_dims - 1)..., capacity))

    resize!(_neighbor_ids, (2^(n_dims - 1) + 1) * capacity)
    mortars.neighbor_ids = unsafe_wrap(Array, pointer(_neighbor_ids),
                                       (2^(n_dims - 1) + 1, capacity))

    resize!(_node_indices, 2 * capacity)
    mortars.node_indices = unsafe_wrap(Array, pointer(_node_indices), (2, capacity))

    return nothing
end

# Create mortar container and initialize mortar data.
function init_mortars(mesh::Union{P4estMesh, T8codeMesh}, equations, basis, elements)
    NDIMS = ndims(elements)
    uEltype = eltype(elements)

    # Initialize container
    n_mortars = count_required_surfaces(mesh).mortars

    _u = Vector{uEltype}(undef,
                         2 * nvariables(equations) * 2^(NDIMS - 1) *
                         nnodes(basis)^(NDIMS - 1) * n_mortars)
    u = unsafe_wrap(Array, pointer(_u),
                    (2, nvariables(equations), 2^(NDIMS - 1),
                     ntuple(_ -> nnodes(basis), NDIMS - 1)..., n_mortars))

    _neighbor_ids = Vector{Int}(undef, (2^(NDIMS - 1) + 1) * n_mortars)
    neighbor_ids = unsafe_wrap(Array, pointer(_neighbor_ids),
                               (2^(NDIMS - 1) + 1, n_mortars))

    _node_indices = Vector{NTuple{NDIMS, Symbol}}(undef, 2 * n_mortars)
    node_indices = unsafe_wrap(Array, pointer(_node_indices), (2, n_mortars))

    mortars = P4estMortarContainer{NDIMS, uEltype, NDIMS + 1, NDIMS + 3}(u,
                                                                         neighbor_ids,
                                                                         node_indices,
                                                                         _u,
                                                                         _neighbor_ids,
                                                                         _node_indices)

    if n_mortars > 0
        init_mortars!(mortars, mesh)
    end

    return mortars
end

function init_mortars!(mortars, mesh::P4estMesh)
    init_surfaces!(nothing, mortars, nothing, mesh)

    return mortars
end

function reinitialize_containers!(mesh::P4estMesh, equations, dg::DGSEM, cache)
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

    # re-initialize mortars container
    if hasproperty(cache, :mortars) # cache_parabolic does not carry mortars
        @unpack mortars = cache
        resize!(mortars, required.mortars)

        # re-initialize containers together to reduce
        # the number of iterations over the mesh in `p4est`
        init_surfaces!(interfaces, mortars, boundaries, mesh)
    else
        init_surfaces!(interfaces, nothing, boundaries, mesh)
    end
end

# A helper struct used in initialization methods below
mutable struct InitSurfacesIterFaceUserData{Interfaces, Mortars, Boundaries, Mesh}
    interfaces::Interfaces
    interface_id::Int
    mortars::Mortars
    mortar_id::Int
    boundaries::Boundaries
    boundary_id::Int
    mesh::Mesh
end

function InitSurfacesIterFaceUserData(interfaces, mortars, boundaries, mesh)
    return InitSurfacesIterFaceUserData{typeof(interfaces), typeof(mortars),
                                        typeof(boundaries), typeof(mesh)}(interfaces, 1,
                                                                          mortars, 1,
                                                                          boundaries, 1,
                                                                          mesh)
end

function init_surfaces_iter_face(info, user_data)
    # Unpack user_data
    data = unsafe_pointer_to_objref(Ptr{InitSurfacesIterFaceUserData}(user_data))

    # Function barrier because the unpacked user_data above is type-unstable
    init_surfaces_iter_face_inner(info, data)
end

# 2D
function cfunction(::typeof(init_surfaces_iter_face), ::Val{2})
    @cfunction(init_surfaces_iter_face, Cvoid,
               (Ptr{p4est_iter_face_info_t}, Ptr{Cvoid}))
end
# 3D
function cfunction(::typeof(init_surfaces_iter_face), ::Val{3})
    @cfunction(init_surfaces_iter_face, Cvoid,
               (Ptr{p8est_iter_face_info_t}, Ptr{Cvoid}))
end

# Function barrier for type stability
function init_surfaces_iter_face_inner(info, user_data)
    @unpack interfaces, mortars, boundaries = user_data
    info_pw = PointerWrapper(info)
    elem_count = info_pw.sides.elem_count[]

    if elem_count == 2
        # Two neighboring elements => Interface or mortar

        # Extract surface data
        sides_pw = (load_pointerwrapper_side(info_pw, 1),
                    load_pointerwrapper_side(info_pw, 2))

        if sides_pw[1].is_hanging[] == false && sides_pw[2].is_hanging[] == false
            # No hanging nodes => normal interface
            if interfaces !== nothing
                init_interfaces_iter_face_inner(info_pw, sides_pw, user_data)
            end
        else
            # Hanging nodes => mortar
            if mortars !== nothing
                init_mortars_iter_face_inner(info_pw, sides_pw, user_data)
            end
        end
    elseif elem_count == 1
        # One neighboring elements => boundary
        if boundaries !== nothing
            init_boundaries_iter_face_inner(info_pw, user_data)
        end
    end
    return nothing
end

function init_surfaces!(interfaces, mortars, boundaries, mesh::P4estMesh)
    # Let `p4est` iterate over all interfaces and call init_surfaces_iter_face
    iter_face_c = cfunction(init_surfaces_iter_face, Val(ndims(mesh)))
    user_data = InitSurfacesIterFaceUserData(interfaces, mortars, boundaries, mesh)

    iterate_p4est(mesh.p4est, user_data; iter_face_c = iter_face_c)

    return interfaces
end

# Initialization of interfaces after the function barrier
function init_interfaces_iter_face_inner(info_pw, sides_pw, user_data)
    @unpack interfaces, interface_id, mesh = user_data
    user_data.interface_id += 1

    # Get Tuple of local trees, one-based indexing
    trees_pw = (load_pointerwrapper_tree(mesh.p4est, sides_pw[1].treeid[] + 1),
                load_pointerwrapper_tree(mesh.p4est, sides_pw[2].treeid[] + 1))
    # Quadrant numbering offsets of the quadrants at this interface
    offsets = SVector(trees_pw[1].quadrants_offset[],
                      trees_pw[2].quadrants_offset[])

    local_quad_ids = SVector(sides_pw[1].is.full.quadid[], sides_pw[2].is.full.quadid[])
    # Global IDs of the neighboring quads
    quad_ids = offsets + local_quad_ids

    # Write data to interfaces container
    # `p4est` uses zero-based indexing; convert to one-based indexing
    interfaces.neighbor_ids[1, interface_id] = quad_ids[1] + 1
    interfaces.neighbor_ids[2, interface_id] = quad_ids[2] + 1

    # Face at which the interface lies
    faces = (sides_pw[1].face[], sides_pw[2].face[])

    # Save interfaces.node_indices dimension specific in containers_[23]d.jl
    init_interface_node_indices!(interfaces, faces, info_pw.orientation[], interface_id)

    return nothing
end

# Initialization of boundaries after the function barrier
function init_boundaries_iter_face_inner(info_pw, user_data)
    @unpack boundaries, boundary_id, mesh = user_data
    user_data.boundary_id += 1

    # Extract boundary data
    side_pw = load_pointerwrapper_side(info_pw)
    # Get local tree, one-based indexing
    tree_pw = load_pointerwrapper_tree(mesh.p4est, side_pw.treeid[] + 1)
    # Quadrant numbering offset of this quadrant
    offset = tree_pw.quadrants_offset[]

    # Verify before accessing is.full, but this should never happen
    @assert side_pw.is_hanging[] == false

    local_quad_id = side_pw.is.full.quadid[]
    # Global ID of this quad
    quad_id = offset + local_quad_id

    # Write data to boundaries container
    # `p4est` uses zero-based indexing; convert to one-based indexing
    boundaries.neighbor_ids[boundary_id] = quad_id + 1

    # Face at which the boundary lies
    face = side_pw.face[]

    # Save boundaries.node_indices dimension specific in containers_[23]d.jl
    init_boundary_node_indices!(boundaries, face, boundary_id)

    # One-based indexing
    boundaries.name[boundary_id] = mesh.boundary_names[face + 1, side_pw.treeid[] + 1]

    return nothing
end

# Initialization of mortars after the function barrier
function init_mortars_iter_face_inner(info_pw, sides_pw, user_data)
    @unpack mortars, mortar_id, mesh = user_data
    user_data.mortar_id += 1

    # Get Tuple of local trees, one-based indexing
    trees_pw = (load_pointerwrapper_tree(mesh.p4est, sides_pw[1].treeid[] + 1),
                load_pointerwrapper_tree(mesh.p4est, sides_pw[2].treeid[] + 1))
    # Quadrant numbering offsets of the quadrants at this interface
    offsets = SVector(trees_pw[1].quadrants_offset[],
                      trees_pw[2].quadrants_offset[])

    if sides_pw[1].is_hanging[] == true
        # Left is small, right is large
        faces = (sides_pw[1].face[], sides_pw[2].face[])

        local_small_quad_ids = sides_pw[1].is.hanging.quadid[]
        # Global IDs of the two small quads
        small_quad_ids = offsets[1] .+ local_small_quad_ids

        # Just be sure before accessing is.full
        @assert sides_pw[2].is_hanging[] == false
        large_quad_id = offsets[2] + sides_pw[2].is.full.quadid[]
    else # sides_pw[2].is_hanging[] == true
        # Right is small, left is large.
        # init_mortar_node_indices! below expects side 1 to contain the small elements.
        faces = (sides_pw[2].face[], sides_pw[1].face[])

        local_small_quad_ids = sides_pw[2].is.hanging.quadid[]
        # Global IDs of the two small quads
        small_quad_ids = offsets[2] .+ local_small_quad_ids

        # Just be sure before accessing is.full
        @assert sides_pw[1].is_hanging[] == false
        large_quad_id = offsets[1] + sides_pw[1].is.full.quadid[]
    end

    # Write data to mortar container, 1 and 2 are the small elements
    # `p4est` uses zero-based indexing; convert to one-based indexing
    mortars.neighbor_ids[1:(end - 1), mortar_id] .= small_quad_ids[:] .+ 1
    # Last entry is the large element
    mortars.neighbor_ids[end, mortar_id] = large_quad_id + 1

    init_mortar_node_indices!(mortars, faces, info_pw.orientation[], mortar_id)

    return nothing
end

# Iterate over all interfaces and count
# - (inner) interfaces
# - mortars
# - boundaries
# and collect the numbers in `user_data` in this order.
function count_surfaces_iter_face(info, user_data)
    info_pw = PointerWrapper(info)
    elem_count = info_pw.sides.elem_count[]

    if elem_count == 2
        # Two neighboring elements => Interface or mortar

        # Extract surface data
        sides_pw = (load_pointerwrapper_side(info_pw, 1),
                    load_pointerwrapper_side(info_pw, 2))

        if sides_pw[1].is_hanging[] == false && sides_pw[2].is_hanging[] == false
            # No hanging nodes => normal interface
            # Unpack user_data = [interface_count] and increment interface_count
            pw = PointerWrapper(Int, user_data)
            id = pw[1]
            pw[1] = id + 1
        else
            # Hanging nodes => mortar
            # Unpack user_data = [mortar_count] and increment mortar_count
            pw = PointerWrapper(Int, user_data)
            id = pw[2]
            pw[2] = id + 1
        end
    elseif elem_count == 1
        # One neighboring elements => boundary

        # Unpack user_data = [boundary_count] and increment boundary_count
        pw = PointerWrapper(Int, user_data)
        id = pw[3]
        pw[3] = id + 1
    end

    return nothing
end

# 2D
function cfunction(::typeof(count_surfaces_iter_face), ::Val{2})
    @cfunction(count_surfaces_iter_face, Cvoid,
               (Ptr{p4est_iter_face_info_t}, Ptr{Cvoid}))
end
# 3D
function cfunction(::typeof(count_surfaces_iter_face), ::Val{3})
    @cfunction(count_surfaces_iter_face, Cvoid,
               (Ptr{p8est_iter_face_info_t}, Ptr{Cvoid}))
end

function count_required_surfaces(mesh::P4estMesh)
    # Let `p4est` iterate over all interfaces and call count_surfaces_iter_face
    iter_face_c = cfunction(count_surfaces_iter_face, Val(ndims(mesh)))

    # interfaces, mortars, boundaries
    user_data = [0, 0, 0]

    iterate_p4est(mesh.p4est, user_data; iter_face_c = iter_face_c)

    # Return counters
    return (interfaces = user_data[1],
            mortars = user_data[2],
            boundaries = user_data[3])
end

# Return direction of the face, which is indexed by node_indices
@inline function indices2direction(indices)
    if indices[1] === :begin
        return 1
    elseif indices[1] === :end
        return 2
    elseif indices[2] === :begin
        return 3
    elseif indices[2] === :end
        return 4
    elseif indices[3] === :begin
        return 5
    else # if indices[3] === :end
        return 6
    end
end

include("containers_2d.jl")
include("containers_3d.jl")
include("containers_parallel.jl")
include("containers_parallel_2d.jl")
include("containers_parallel_3d.jl")
end # @muladd
