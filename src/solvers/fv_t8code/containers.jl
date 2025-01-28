# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

function reinitialize_containers!(mesh::T8codeMesh, equations, solver::FV, cache)
    # Re-initialize elements container.
    @unpack elements = cache
    resize!(elements, ncells(mesh))
    init_elements!(elements, mesh, equations, solver)

    count_required_surfaces!(mesh)
    # Resize interfaces container.
    @unpack interfaces = cache
    resize!(interfaces, mesh.ninterfaces + mesh.nmpiinterfaces)

    # Resize mortars container.
    @unpack mortars = cache
    resize!(mortars, mesh.nmortars + mesh.nmpimortars)

    # Resize boundaries container.
    @unpack boundaries = cache
    resize!(boundaries, mesh.nboundaries)

    fill_mesh_info_fv!(mesh, interfaces, boundaries, mortars,
                       mesh.boundary_names)

    (; solution_data, domain_data, gradient_data) = cache.communication_data
    num_ghost_elements = t8_forest_get_num_ghosts(mesh.forest)
    resize!(solution_data, ncells(mesh) + num_ghost_elements)
    resize!(domain_data, ncells(mesh) + num_ghost_elements)
    resize!(gradient_data, ncells(mesh) + num_ghost_elements)
    exchange_domain_data!(cache.communication_data, elements, mesh, equations, solver)

    # Reinitialize reconstruction stencil
    if !solver.extended_reconstruction_stencil
        init_reconstruction_stencil!(elements, interfaces, boundaries, mortars,
                                     cache.communication_data, mesh, equations, solver)
    end

    return nothing
end

# Container data structure (structure-of-arrays style) for DG elements
mutable struct T8codeFVElementContainer{NDIMS, RealT <: Real, uEltype <: Real}
    level::Vector{Cint}     # [element]
    volume::Vector{RealT}   # [element]
    midpoint::Matrix{RealT} # [dimension, element]
    dx::Vector{RealT}       # [element] - characteristic length (for CFL condition).

    num_faces::Vector{Cint}         # [element]
    face_midpoints::Array{RealT, 3} # [dimension, face, element]
    face_areas::Matrix{RealT}       # [face, element]
    face_normals::Array{RealT, 3}   # [dimension, face, element]

    reconstruction_stencil::Vector{Vector{Int}}                     # Reconstruction stencil vector with neighbors per [element]
    reconstruction_distance::Vector{Vector{SVector{NDIMS, RealT}}}  # Reconstruction stencil vector with distances per [element]
    reconstruction_corner_elements::Array{Vector{Int}, 2}           # Reconstruction stencil array with neighbor elements at corner per [corner, element]
    reconstruction_gradient::Array{RealT, 4}            # [dimension, variable, slope_stencils, element], slope_stencil: i = use all neighbors exclusive i, n_neighbors = use all neighbors
    reconstruction_gradient_limited::Array{RealT, 3}    # [dimension, variable, element]

    # internal `resize!`able storage
    _midpoint::Vector{RealT}
    _face_midpoints::Vector{RealT}
    _face_areas::Vector{RealT}
    _face_normals::Vector{RealT}
    _reconstruction_corner_elements::Vector{Vector{Int}}
    _reconstruction_gradient::Vector{RealT}
    _reconstruction_gradient_limited::Vector{RealT}
end

@inline Base.ndims(::T8codeFVElementContainer{NDIMS}) where {NDIMS} = NDIMS
@inline nelements(elements::T8codeFVElementContainer) = length(elements.num_faces)
@inline function Base.eltype(::T8codeFVElementContainer{NDIMS, RealT, uEltype}) where {
                                                                                       NDIMS,
                                                                                       RealT,
                                                                                       uEltype
                                                                                       }
    uEltype
end

@inline is_ghost_cell(element, mesh) = element > ncells(mesh)

# See explanation of Base.resize! for the element container
function Base.resize!(elements::T8codeFVElementContainer, capacity)
    (; _midpoint, _face_midpoints, _face_areas, _face_normals, _reconstruction_gradient, _reconstruction_gradient_limited, _reconstruction_corner_elements) = elements

    n_dims = ndims(elements)
    n_variables = size(elements.reconstruction_gradient, 2)
    max_number_faces = size(elements.face_midpoints, 2)

    resize!(elements.level, capacity)
    resize!(elements.volume, capacity)
    resize!(elements.dx, capacity)
    resize!(elements.num_faces, capacity)

    resize!(_midpoint, n_dims * capacity)
    elements.midpoint = unsafe_wrap(Array, pointer(_midpoint), (n_dims, capacity))

    resize!(_face_midpoints, n_dims * max_number_faces * capacity)
    elements.face_midpoints = unsafe_wrap(Array, pointer(_face_midpoints),
                                          (n_dims, max_number_faces, capacity))

    resize!(_face_areas, max_number_faces * capacity)
    elements.face_areas = unsafe_wrap(Array, pointer(_face_areas),
                                      (max_number_faces, capacity))

    resize!(_face_normals, n_dims * max_number_faces * capacity)
    elements.face_normals = unsafe_wrap(Array, pointer(_face_normals),
                                        (n_dims, max_number_faces, capacity))

    resize!(elements.reconstruction_stencil, capacity)
    resize!(elements.reconstruction_distance, capacity)

    resize!(_reconstruction_corner_elements, max_number_faces * capacity)
    elements.reconstruction_corner_elements = unsafe_wrap(Array,
                                                          pointer(_reconstruction_corner_elements),
                                                          (max_number_faces, capacity))

    max_neighbors_per_face = 2^(n_dims - 1)
    resize!(_reconstruction_gradient,
            n_dims * n_variables * (max_neighbors_per_face * max_number_faces + 1) *
            capacity)
    elements.reconstruction_gradient = unsafe_wrap(Array,
                                                   pointer(_reconstruction_gradient),
                                                   (n_dims, n_variables,
                                                    max_neighbors_per_face *
                                                    max_number_faces + 1, capacity))

    resize!(_reconstruction_gradient_limited, n_dims * n_variables * capacity)
    elements.reconstruction_gradient_limited = unsafe_wrap(Array,
                                                           pointer(_reconstruction_gradient_limited),
                                                           (n_dims, n_variables,
                                                            capacity))

    return nothing
end

# Create element container and initialize element data
function init_elements(mesh::T8codeMesh{NDIMS, RealT},
                       equations,
                       solver::FV,
                       ::Type{uEltype}) where {NDIMS, RealT <: Real, uEltype <: Real}
    (; forest) = mesh
    # Check that the forest is a committed.
    @assert(t8_forest_is_committed(forest)==1)

    nelements = ncells(mesh)
    n_variables = nvariables(equations)
    (; max_number_faces) = mesh

    level = Vector{Cint}(undef, nelements)
    volume = Vector{RealT}(undef, nelements)
    dx = Vector{RealT}(undef, nelements)
    num_faces = Vector{Cint}(undef, nelements)

    _midpoint = Vector{RealT}(undef, NDIMS * nelements)
    midpoint = unsafe_wrap(Array, pointer(_midpoint), (NDIMS, nelements))

    _face_midpoints = Vector{RealT}(undef, NDIMS * max_number_faces * nelements)
    face_midpoints = unsafe_wrap(Array, pointer(_face_midpoints),
                                 (NDIMS, max_number_faces, nelements))

    _face_areas = Vector{RealT}(undef, max_number_faces * nelements)
    face_areas = unsafe_wrap(Array, pointer(_face_areas), (max_number_faces, nelements))

    _face_normals = Vector{RealT}(undef, NDIMS * max_number_faces * nelements)
    face_normals = unsafe_wrap(Array, pointer(_face_normals),
                               (NDIMS, max_number_faces, nelements))

    reconstruction_stencil = Vector{Vector{Int}}(undef, nelements)
    reconstruction_distance = Vector{Vector{SVector{NDIMS, RealT}}}(undef, nelements)

    _reconstruction_corner_elements = Vector{Vector{Int}}(undef,
                                                          max_number_faces * nelements)
    reconstruction_corner_elements = unsafe_wrap(Array,
                                                 pointer(_reconstruction_corner_elements),
                                                 (max_number_faces, nelements))

    _reconstruction_gradient = Vector{RealT}(undef,
                                             NDIMS * n_variables *
                                             (max_number_faces + 1) * nelements)
    reconstruction_gradient = unsafe_wrap(Array, pointer(_reconstruction_gradient),
                                          (NDIMS, n_variables, max_number_faces + 1,
                                           nelements))

    _reconstruction_gradient_limited = Vector{RealT}(undef,
                                                     NDIMS * n_variables * nelements)
    reconstruction_gradient_limited = unsafe_wrap(Array,
                                                  pointer(_reconstruction_gradient_limited),
                                                  (NDIMS, n_variables, nelements))

    elements = T8codeFVElementContainer{NDIMS, RealT, uEltype}(level, volume, midpoint,
                                                               dx, num_faces,
                                                               face_midpoints,
                                                               face_areas, face_normals,
                                                               reconstruction_stencil,
                                                               reconstruction_distance,
                                                               reconstruction_corner_elements,
                                                               reconstruction_gradient,
                                                               reconstruction_gradient_limited,
                                                               _midpoint,
                                                               _face_midpoints,
                                                               _face_areas,
                                                               _face_normals,
                                                               _reconstruction_corner_elements,
                                                               _reconstruction_gradient,
                                                               _reconstruction_gradient_limited)

    init_elements!(elements, mesh, equations, solver)

    return elements
end

function init_elements!(elements, mesh::T8codeMesh, equations, solver::FV)
    (; forest) = mesh

    n_dims = ndims(mesh)
    (; level, volume, dx, num_faces, face_areas, face_midpoints, face_normals) = elements

    midpoint = Vector{Cdouble}(undef, n_dims)

    face_midpoint = Vector{Cdouble}(undef, 3) # Need NDIMS=3 for t8code API
    face_normal = Vector{Cdouble}(undef, 3) # Need NDIMS=3 for t8code API
    corners = Array{Cdouble, 3}(undef, n_dims, mesh.max_number_faces, length(volume))
    corner_node = Vector{Cdouble}(undef, n_dims)

    num_local_trees = t8_forest_get_num_local_trees(forest)

    # Loop over all local trees in the forest.
    current_index = 0
    for itree in 0:(num_local_trees - 1)
        tree_class = t8_forest_get_tree_class(forest, itree)
        eclass_scheme = t8_forest_get_eclass_scheme(forest, tree_class)

        # Get the number of elements of this tree.
        num_elements_in_tree = t8_forest_get_tree_num_elements(forest, itree)

        # Loop over all local elements in the tree.
        for ielement in 0:(num_elements_in_tree - 1)
            current_index += 1 # Note: Julia has 1-based indexing, while C/C++ starts with 0.

            element = t8_forest_get_element_in_tree(forest, itree, ielement)

            level[current_index] = t8_element_level(eclass_scheme, element)
            volume[current_index] = t8_forest_element_volume(forest, itree, element)

            t8_forest_element_centroid(forest, itree, element, pointer(midpoint))
            for dim in 1:n_dims
                elements.midpoint[dim, current_index] = midpoint[dim]
            end

            # Characteristic length of the element. It is an approximation since only
            # for the element type `lines` this can be exact.
            dx[current_index] = t8_forest_element_diam(forest, itree, element)

            # Loop over all faces of an element.
            num_faces[current_index] = t8_element_num_faces(eclass_scheme, element)

            for iface in 1:num_faces[current_index]
                # C++ is zero-indexed
                t8_forest_element_coordinate(forest, itree, element, iface - 1,
                                             @views(corner_node))

                t8_forest_element_face_centroid(forest, itree, element, iface - 1,
                                                face_midpoint)

                face_area = t8_forest_element_face_area(forest, itree, element,
                                                        iface - 1)
                face_areas[iface, current_index] = face_area

                t8_forest_element_face_normal(forest, itree, element, iface - 1,
                                              face_normal)
                for dim in 1:n_dims
                    corners[dim, iface, current_index] = corner_node[dim]
                    face_midpoints[dim, iface, current_index] = face_midpoint[dim]
                    face_normals[dim, iface, current_index] = face_normal[dim]
                end
            end
        end
    end

    # Init stencil for reconstruction
    if solver.extended_reconstruction_stencil
        init_extended_reconstruction_stencil!(corners, elements, equations, solver)
    end

    return nothing
end

@inline function init_extended_reconstruction_stencil!(corners, elements, equations,
                                                       solver::FV)
    if solver.order != 2
        return nothing
    end
    (; reconstruction_stencil, reconstruction_distance, reconstruction_corner_elements) = elements
    (; volume, num_faces) = elements

    # Create empty vectors for every element
    for element in eachindex(volume)
        reconstruction_stencil[element] = Int[]
        reconstruction_distance[element] = SVector{eltype(reconstruction_distance)}[]
        for corner in 1:num_faces[element]
            reconstruction_corner_elements[corner, element] = Int[]
        end
    end

    # Add all stencil neighbors to list; including doubled elements
    for element in eachindex(volume)
        # loop over all elements with higher index
        for possible_stencil_neighbor in (element + 1):length(volume)
            # loop over all corners of `element`
            for corner in 1:num_faces[element]
                corner_coords = get_node_coords(corners, equations, solver, corner,
                                                element)
                # loop over all corners of `possible_stencil_neighbor`
                for possible_corner in 1:num_faces[possible_stencil_neighbor]
                    possible_corner_coords = get_node_coords(corners, equations, solver,
                                                             possible_corner,
                                                             possible_stencil_neighbor)
                    if corner_coords == possible_corner_coords
                        neighbor = possible_stencil_neighbor

                        midpoint_element = get_node_coords(elements.midpoint, equations,
                                                           solver, element)
                        midpoint_neighbor = get_node_coords(elements.midpoint,
                                                            equations, solver, neighbor)

                        distance = midpoint_neighbor .- midpoint_element
                        append!(reconstruction_stencil[element], neighbor)
                        push!(reconstruction_distance[element], distance)
                        append!(reconstruction_stencil[neighbor], element)
                        push!(reconstruction_distance[neighbor], -distance)
                        append!(reconstruction_corner_elements[corner, element],
                                neighbor)
                        append!(reconstruction_corner_elements[possible_corner,
                                                               neighbor], element)

                        # elseif # TODO: Handle periodic boundaries; Something like:
                        #     distance = (face_midpoint_element .- midpoint_element) .+
                        #                (midpoint_neighbor .- face_midpoint_neighbor)
                    end
                end
            end
        end
    end

    # Remove all doubled elements from vectors
    for element in eachindex(volume)
        for i in length(reconstruction_stencil[element]):-1:1
            neighbor = reconstruction_stencil[element][i]
            if neighbor in reconstruction_stencil[element][1:(i - 1)]
                popat!(reconstruction_stencil[element], i)
                popat!(reconstruction_distance[element], i)
            end
        end
    end

    for element in eachindex(volume)
        for corner in 1:num_faces[element]
            for i in length(reconstruction_corner_elements[corner, element]):-1:1
                neighbor = reconstruction_corner_elements[corner, element][i]
                if neighbor in reconstruction_corner_elements[corner, element][1:(i - 1)]
                    popat!(reconstruction_corner_elements[corner, element], i)
                end
            end
        end
    end

    return nothing
end

function init_reconstruction_stencil!(elements, interfaces, boundaries, mortars,
                                      communication_data,
                                      mesh, equations, solver::FV)
    if solver.order != 2 # type instability?
        return nothing
    end
    (; reconstruction_stencil, reconstruction_distance) = elements
    (; domain_data) = communication_data

    # Create empty vectors for every element
    for element in eachindex(reconstruction_stencil)
        reconstruction_stencil[element] = Int[]
        reconstruction_distance[element] = SVector{eltype(reconstruction_distance)}[]
    end

    (; neighbor_ids, faces) = interfaces
    for interface in axes(neighbor_ids, 2)
        element1 = neighbor_ids[1, interface]
        element2 = neighbor_ids[2, interface]
        face_element1 = faces[1, interface]
        face_element2 = faces[2, interface]

        midpoint_element1 = domain_data[element1].midpoint
        midpoint_element2 = domain_data[element2].midpoint
        face_midpoint_element1 = domain_data[element1].face_midpoints[face_element1]
        face_midpoint_element2 = domain_data[element2].face_midpoints[face_element2]

        # TODO: How to handle periodic boundaries?
        if isapprox(face_midpoint_element1, face_midpoint_element2)
            distance = midpoint_element2 .- midpoint_element1
        else
            distance = (face_midpoint_element1 .- midpoint_element1) .+
                       (midpoint_element2 .- face_midpoint_element2)
        end
        append!(reconstruction_stencil[element1], element2)
        push!(reconstruction_distance[element1], distance)
        # only if element2 is local element
        if !is_ghost_cell(element2, mesh)
            append!(reconstruction_stencil[element2], element1)
            push!(reconstruction_distance[element2], -distance)
        end
    end

    (; neighbor_ids, faces, n_local_elements_small) = mortars
    for mortar in axes(neighbor_ids, 2)
        element_large = neighbor_ids[end, mortar]
        face_element_large = faces[end, mortar]

        midpoint_element_large = domain_data[element_large].midpoint

        n_positions = n_local_elements_small[mortar]
        for position in 1:n_positions
            element_small = neighbor_ids[position, mortar]
            face_element_small = faces[position, mortar]
            midpoint_element_small = domain_data[element_small].midpoint
            face_midpoint_element_small = domain_data[element_small].face_midpoints[face_element_small]

            # TODO: The face midpoint of the large element is not the correct here.
            face_midpoint_element_large = domain_data[element_large].face_midpoints[face_element_large]

            # TODO: How to handle periodic boundaries?
            # Hacky solution to figure out if mortar is at periodic boundary
            distance_face_large2face_small = sqrt(sum(abs.(face_midpoint_element_large .-
                                                           face_midpoint_element_small) .^
                                                      2))
            distance_face_large2mid_large = sqrt(sum(abs.(face_midpoint_element_large .-
                                                          midpoint_element_small) .^ 2))
            # Not at periodic boundary
            if distance_face_large2face_small < distance_face_large2mid_large
                distance = midpoint_element_small .- midpoint_element_large
            else # At periodic boundary
                # TODO: See above
                distance = (face_midpoint_element_large .- midpoint_element_large) .+
                           (midpoint_element_small .- face_midpoint_element_small)
            end

            # Add info only if element is local element
            if !is_ghost_cell(element_large, mesh)
                append!(reconstruction_stencil[element_large], element_small)
                push!(reconstruction_distance[element_large], distance)
            end
            if !is_ghost_cell(element_small, mesh)
                append!(reconstruction_stencil[element_small], element_large)
                push!(reconstruction_distance[element_small], -distance)
            end
        end
    end

    return nothing
end

# Container data structure (structure-of-arrays style) for FV interfaces
mutable struct T8codeFVInterfaceContainer{uEltype <: Real} <: AbstractContainer
    u::Array{uEltype, 3}                # [primary/secondary, variable, interface]
    neighbor_ids::Matrix{Int}           # [primary/secondary, interface]
    faces::Matrix{Int}                  # [primary/secondary, interface]

    # internal `resize!`able storage
    _u::Vector{uEltype}
    _neighbor_ids::Vector{Int}
    _faces::Vector{Int}
end

@inline function ninterfaces(interfaces::T8codeFVInterfaceContainer)
    size(interfaces.neighbor_ids, 2)
end

# See explanation of Base.resize! for the element container
function Base.resize!(interfaces::T8codeFVInterfaceContainer, capacity)
    (; _u, _neighbor_ids, _faces) = interfaces

    n_variables = size(interfaces.u, 2)

    resize!(_u, 2 * n_variables * capacity)
    interfaces.u = unsafe_wrap(Array, pointer(_u),
                               (2, n_variables, capacity))

    resize!(_neighbor_ids, 2 * capacity)
    interfaces.neighbor_ids = unsafe_wrap(Array, pointer(_neighbor_ids), (2, capacity))

    resize!(_faces, 2 * capacity)
    interfaces.faces = unsafe_wrap(Array, pointer(_faces), (2, capacity))

    return nothing
end

# Create interface container and initialize interface data.
function init_interfaces(mesh::T8codeMesh, equations, solver::FV, uEltype)
    # Initialize container
    n_interfaces = count_required_surfaces(mesh).interfaces
    if mpi_parallel(mesh) == true
        n_interfaces += count_required_surfaces(mesh).mpi_interfaces
    end

    _u = Vector{uEltype}(undef,
                         2 * nvariables(equations) * n_interfaces)
    u = unsafe_wrap(Array, pointer(_u),
                    (2, nvariables(equations), n_interfaces))

    _neighbor_ids = Vector{Int}(undef, 2 * n_interfaces)
    neighbor_ids = unsafe_wrap(Array, pointer(_neighbor_ids), (2, n_interfaces))

    _faces = Vector{Int}(undef, 2 * n_interfaces)
    faces = unsafe_wrap(Array, pointer(_faces), (2, n_interfaces))

    interfaces = T8codeFVInterfaceContainer{uEltype}(u, neighbor_ids, faces,
                                                     _u, _neighbor_ids, _faces)

    return interfaces
end

mutable struct T8codeFVBoundaryContainer{uEltype <: Real} <: AbstractContainer
    u::Array{uEltype, 2}      # [variable, boundary]
    neighbor_ids::Vector{Int} # [boundary]
    faces::Vector{Int}        # [boundary]
    name::Vector{Symbol}      # [boundary]

    # internal `resize!`able storage
    _u::Vector{uEltype}
end

@inline function nboundaries(boundaries::T8codeFVBoundaryContainer)
    length(boundaries.neighbor_ids)
end

# See explanation of Base.resize! for the element container
function Base.resize!(boundaries::T8codeFVBoundaryContainer, capacity)
    (; _u, neighbor_ids, faces, name) = boundaries

    n_variables = size(boundaries.u, 1)

    resize!(_u, n_variables * capacity)
    boundaries.u = unsafe_wrap(Array, pointer(_u),
                               (n_variables, capacity))

    resize!(neighbor_ids, capacity)

    resize!(faces, capacity)

    resize!(name, capacity)

    return nothing
end

# Create interface container and initialize interface data in `elements`.
function init_boundaries(mesh::T8codeMesh, equations, solver::FV, uEltype)
    # Initialize container
    n_boundaries = count_required_surfaces(mesh).boundaries

    _u = Vector{uEltype}(undef,
                         nvariables(equations) * n_boundaries)
    u = unsafe_wrap(Array, pointer(_u),
                    (nvariables(equations), n_boundaries))

    neighbor_ids = Vector{Int}(undef, n_boundaries)
    faces = Vector{Int}(undef, n_boundaries)
    names = Vector{Symbol}(undef, n_boundaries)

    boundaries = T8codeFVBoundaryContainer{uEltype}(u, neighbor_ids, faces, names, _u)

    return boundaries
end

mutable struct T8codeFVMortarContainer{NDIMS, uEltype <: Real} <: AbstractContainer
    u::Array{uEltype, 4}                # [small/large side, variable, position, mortar]
    neighbor_ids::Matrix{Int}           # [position, mortar]
    faces::Matrix{Int}                  # [position, mortar]
    n_local_elements_small::Vector{Int} # [mortar]

    # internal `resize!`able storage
    _u::Vector{uEltype}
    _neighbor_ids::Vector{Int}
    _faces::Vector{Int}
end

@inline nmortars(mortars::T8codeFVMortarContainer) = size(mortars.neighbor_ids, 2)
@inline Base.ndims(::T8codeFVMortarContainer{NDIMS}) where {NDIMS} = NDIMS

# See explanation of Base.resize! for the element container
function Base.resize!(mortars::T8codeFVMortarContainer, capacity)
    @unpack _u, _neighbor_ids, _faces = mortars

    n_dims = ndims(mortars)
    n_variables = size(mortars.u, 2)

    resize!(_u, 2 * n_variables * 2^(n_dims - 1) * capacity)
    mortars.u = unsafe_wrap(Array, pointer(_u),
                            (2, n_variables, 2^(n_dims - 1), capacity))

    resize!(_neighbor_ids, (2^(n_dims - 1) + 1) * capacity)
    mortars.neighbor_ids = unsafe_wrap(Array, pointer(_neighbor_ids),
                                       (2^(n_dims - 1) + 1, capacity))

    resize!(_faces, (2^(n_dims - 1) + 1) * capacity)
    mortars.faces = unsafe_wrap(Array, pointer(_faces),
                                (2^(n_dims - 1) + 1, capacity))

    resize!(mortars.n_local_elements_small, capacity)

    return nothing
end

# Create mortar container and initialize mortar data.
function init_mortars(mesh::T8codeMesh, equations, solver::FV, uEltype)
    NDIMS = ndims(mesh)

    # Initialize container
    n_mortars = count_required_surfaces(mesh).mortars
    if mpi_parallel(mesh) == true
        n_mortars += count_required_surfaces(mesh).mpi_mortars
    end

    _u = Vector{uEltype}(undef,
                         2 * nvariables(equations) * 2^(NDIMS - 1) * n_mortars)
    u = unsafe_wrap(Array, pointer(_u),
                    (2, nvariables(equations), 2^(NDIMS - 1), n_mortars))

    _neighbor_ids = Vector{Int}(undef, (2^(NDIMS - 1) + 1) * n_mortars)
    neighbor_ids = unsafe_wrap(Array, pointer(_neighbor_ids),
                               (2^(NDIMS - 1) + 1, n_mortars))

    _faces = Vector{NTuple{NDIMS, Symbol}}(undef, (2^(NDIMS - 1) + 1) * n_mortars)
    faces = unsafe_wrap(Array, pointer(_faces),
                        (2^(NDIMS - 1) + 1, n_mortars))

    n_local_elements_small = Vector{Int}(undef, n_mortars)

    mortars = T8codeFVMortarContainer{NDIMS, uEltype}(u, neighbor_ids, faces,
                                                      _u, _neighbor_ids, _faces,
                                                      n_local_elements_small)

    return mortars
end

function init_communication_data!(mesh::T8codeMesh, equations)
    (; forest) = mesh
    # Check that the forest is a committed.
    @assert(t8_forest_is_committed(forest)==1)

    # Get the number of local elements of forest.
    num_local_elements = ncells(mesh)
    # Get the number of ghost elements of forest.
    num_ghost_elements = t8_forest_get_num_ghosts(forest)

    # Build an array of our data that is as long as the number of elements plus
    # the number of ghosts.
    solution_data = Vector{T8codeSolutionContainer{nvariables(equations)}}(undef,
                                                                           num_local_elements +
                                                                           num_ghost_elements)

    domain_data = Vector{T8codeReconstructionContainer{ndims(equations),
                                                       mesh.max_number_faces}}(undef,
                                                                               num_local_elements +
                                                                               num_ghost_elements)

    gradient_data = Vector{T8codeGradientContainer{ndims(equations),
                                                   nvariables(equations)}}(undef,
                                                                           num_local_elements +
                                                                           num_ghost_elements)

    return (; solution_data, domain_data, gradient_data)
end

# Each process has computed the data entries for its local elements. In order
# to get the values for the ghost elements, we use
# t8_forest_ghost_exchange_data. Calling this function will fill all the ghost
# entries of our element data array with the value on the process that owns the
# corresponding element.
function exchange_ghost_data(mesh, container)
    # t8_forest_ghost_exchange_data expects an sc_array (of length num_local_elements + num_ghosts).
    # We wrap our data array to an sc_array.
    sc_array_wrapper = T8code.Libt8.sc_array_new_data(pointer(container),
                                                      sizeof(typeof(container[1])),
                                                      length(container))

    # Carry out the data exchange. The entries with indices > num_local_elements will get overwritten.
    t8_forest_ghost_exchange_data(mesh.forest, sc_array_wrapper)

    # Destroy the wrapper array. This will not free the data memory since we used sc_array_new_data.
    T8code.Libt8.sc_array_destroy(sc_array_wrapper)
end

struct T8codeSolutionContainer{NVARS}
    u::NTuple{NVARS, Cdouble}

    function T8codeSolutionContainer(u)
        new{length(u)}(u)
    end
end

function exchange_solution_data!(u, mesh, equations, solver, cache)
    (; solution_data) = cache.communication_data
    for element in eachelement(solver, cache)
        solution_data[element] = T8codeSolutionContainer(Tuple(get_node_vars(u,
                                                                             equations,
                                                                             solver,
                                                                             element)))
    end
    exchange_ghost_data(mesh, solution_data)

    return nothing
end

struct T8codeGradientContainer{NDIMS, NVARS}
    reconstruction_gradient_limited::NTuple{NVARS, SVector{NDIMS, Cdouble}}

    function T8codeGradientContainer(reconstruction_gradient_limited)
        new{length(reconstruction_gradient_limited[1]),
            length(reconstruction_gradient_limited)}(reconstruction_gradient_limited)
    end
end

function exchange_gradient_data!(reconstruction_gradient_limited,
                                 mesh, equations, solver, cache)
    (; gradient_data) = cache.communication_data
    for element in eachelement(solver, cache)
        gradient_data[element] = T8codeGradientContainer(ntuple(v -> get_node_coords(reconstruction_gradient_limited,
                                                                                     equations,
                                                                                     solver,
                                                                                     v,
                                                                                     element),
                                                                Val(nvariables(equations))))
    end
    exchange_ghost_data(mesh, gradient_data)

    return nothing
end

struct T8codeReconstructionContainer{NDIMS, NFACES}
    midpoint::SVector{NDIMS, Cdouble}
    face_midpoints::SVector{NFACES, SVector{NDIMS, Cdouble}}
    face_areas::SVector{NFACES, Cdouble}
    face_normals::SVector{NFACES, SVector{NDIMS, Cdouble}}

    function T8codeReconstructionContainer(midpoint, face_midpoints, face_areas,
                                           face_normals)
        new{length(midpoint), length(face_midpoints)}(midpoint, face_midpoints,
                                                      face_areas, face_normals)
    end
end

function exchange_domain_data!(communication_data, elements, mesh, equations, solver)
    (; domain_data) = communication_data
    (; midpoint, face_midpoints, face_areas, face_normals, num_faces) = elements

    n_dims = ndims(equations)
    vec_zero = SVector{n_dims}(zeros(eltype(midpoint), n_dims))
    face_midpoints_ = Vector{typeof(vec_zero)}(undef, mesh.max_number_faces)
    face_normals_ = Vector{typeof(vec_zero)}(undef, mesh.max_number_faces)
    face_areas_ = zeros(eltype(face_areas), mesh.max_number_faces)
    for element in 1:ncells(mesh)
        for face in 1:num_faces[element]
            face_midpoints_[face] = get_node_coords(face_midpoints, equations, solver,
                                                    face, element)
            face_normals_[face] = get_node_coords(face_normals, equations, solver,
                                                  face, element)
            face_areas_[face] = face_areas[face, element]
        end
        domain_data[element] = T8codeReconstructionContainer(get_node_coords(midpoint,
                                                                             equations,
                                                                             solver,
                                                                             element),
                                                             face_midpoints_,
                                                             face_areas_,
                                                             face_normals_)
    end
    exchange_ghost_data(mesh, domain_data)

    return nothing
end
end # @muladd
