# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

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

    reconstruction_stencil::Vector{Vector{Int}} # Vector with elements inside reconstruction stencil per [element]
    reconstruction_gradient::Array{RealT, 3}

    # internal `resize!`able storage
    _midpoint::Vector{RealT}
    _face_midpoints::Vector{RealT}
    _face_areas::Vector{RealT}
    _face_normals::Vector{RealT}
    _reconstruction_gradient::Vector{RealT}
end

@inline Base.ndims(::T8codeFVElementContainer{NDIMS}) where {NDIMS} = NDIMS
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
    (; _midpoint, _face_midpoints, _face_areas, _face_normals, _reconstruction_gradient) = interfaces

    n_dims = ndims(elements)
    n_variables = size(reconstruction_gradient, 2)
    max_number_faces = size(face_midpoints, 2)

    resize!(elements.level, capacity)
    resize!(elements.volume, capacity)
    resize!(elements.dx, capacity)
    resize!(elements.num_faces, capacity)

    resize!(_midpoint, n_dims * capacity)
    elements.midpoint = unsafe_wrap(Array, pointer(_midpoint), (ndims, capacity))

    resize!(_face_midpoints, n_dims * max_number_faces * capacity)
    elements.face_midpoints = unsafe_wrap(Array, pointer(_face_midpoints),
                                          (ndims, max_number_faces, capacity))

    resize!(_face_areas, max_number_faces * capacity)
    elements.face_areas = unsafe_wrap(Array, pointer(_face_areas),
                                      (max_number_faces, capacity))

    resize!(_face_normals, n_dims * max_number_faces * capacity)
    elements.face_normals = unsafe_wrap(Array, pointer(_face_normals),
                                        (ndims, max_number_faces, capacity))

    resize!(elements.reconstruction_stencil, capacity)

    resize!(_reconstruction_gradient, n_dims * n_variables * capacity)
    elements.reconstruction_gradient = unsafe_wrap(Array, pointer(_face_normals), (ndims, n_variables, capacity))

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

    _face_normals = similar(_face_midpoints)
    face_normals = unsafe_wrap(Array, pointer(_face_normals), size(face_midpoints))

    reconstruction_stencil = Vector{Vector{Int}}(undef, nelements)

    _reconstruction_gradient = Vector{RealT}(undef, NDIMS * n_variables * nelements)
    reconstruction_gradient = unsafe_wrap(Array, pointer(_reconstruction_gradient), (NDIMS, n_variables, nelements))

    elements = T8codeFVElementContainer{NDIMS, RealT, uEltype}(level, volume, midpoint,
                                                               dx, num_faces,
                                                               face_midpoints,
                                                               face_areas, face_normals,
                                                               reconstruction_stencil, reconstruction_gradient,
                                                               _midpoint, _face_midpoints,
                                                               _face_areas, _face_normals,
                                                               _reconstruction_gradient)

    init_elements!(elements, mesh, solver)

    return elements
end

function init_elements!(elements, mesh::T8codeMesh, solver::FV)
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
    if solver.order != 2
        return nothing
    end
    init_reconstruction_stencil!(corners, elements)

    return nothing
end

@inline function init_reconstruction_stencil!(corners, elements)
    (; reconstruction_stencil, volume, num_faces) = elements

    # Create empty vectors for every element
    for element in eachindex(volume)
        reconstruction_stencil[element] = []
    end

    # Add all stencil neighbors to list; including doubled elements
    for element in eachindex(volume)
        # loop over all elements with higher index
        for possible_stencil_neighbor in (element + 1):length(volume)
            # loop over all corners of `element`
            for corner in 1:num_faces[element]
                corner_coords = view(corners, :, corner, element)
                # loop over all corners of `possible_stencil_neighbor`
                for possible_corner in 1:num_faces[possible_stencil_neighbor]
                    if corner_coords == view(corners, :, possible_corner, possible_stencil_neighbor)
                        append!(reconstruction_stencil[element], possible_stencil_neighbor)
                        append!(reconstruction_stencil[possible_stencil_neighbor], element)
                    end
                end
            end
        end
    end

    # Remove all doubled elements from vectors
    for element in eachindex(volume)
        reconstruction_stencil[element] = unique(reconstruction_stencil[element])
    end
    # TODO: How to handle periodic boundaries?

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

function init_solution!(mesh::T8codeMesh, equations)
    (; forest) = mesh
    # Check that the forest is a committed.
    @assert(t8_forest_is_committed(forest)==1)

    # Get the number of local elements of forest.
    num_local_elements = ncells(mesh)
    # Get the number of ghost elements of forest.
    num_ghost_elements = t8_forest_get_num_ghosts(forest)

    # Build an array of our data that is as long as the number of elements plus
    # the number of ghosts.
    u_tmp = Vector{T8codeSolutionContainer{nvariables(equations)}}(undef,
                                                                   num_local_elements +
                                                                   num_ghost_elements)

    return u_tmp
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

function get_variable_wrapped(vec, equations, face_or_variable)
    return SVector(ntuple(@inline(idx->vec[ndims(equations) * (face_or_variable - 1) + idx]),
                          Val(ndims(equations))))
end

function exchange_solution!(u, mesh, equations, solver, cache)
    (; u_tmp) = cache
    for element in eachelement(mesh, solver, cache)
        u_tmp[element] = T8codeSolutionContainer(Tuple(get_node_vars(u, equations,
                                                                     solver,
                                                                     element)))
    end
    exchange_ghost_data(mesh, u_tmp)

    return nothing
end
end # @muladd
