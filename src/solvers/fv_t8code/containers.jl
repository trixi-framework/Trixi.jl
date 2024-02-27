# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

# Container data structure (array-of-structures style) for T8codeMesh elements
# The data that we want to store for each element.
struct T8codeElementContainer{NDIMS, RealT <: Real, uEltype <: Real,
                              MAX_NUMBER_FACES, NDIMS_MAX_NUMBER_FACES}
    level    :: Cint
    volume   :: RealT
    midpoint :: NTuple{NDIMS, RealT}
    dx       :: RealT # Characteristic length (for CFL condition).

    num_faces         :: Cint
    face_midpoints    :: NTuple{NDIMS_MAX_NUMBER_FACES, RealT}
    face_areas        :: NTuple{MAX_NUMBER_FACES, RealT}
    face_normals      :: NTuple{NDIMS_MAX_NUMBER_FACES, RealT}
    face_connectivity :: NTuple{MAX_NUMBER_FACES, t8_locidx_t} # ids of the face neighbors
    boundary_name     :: NTuple{MAX_NUMBER_FACES, Symbol}
    neighbor_faces    :: NTuple{MAX_NUMBER_FACES, t8_locidx_t}

    function T8codeElementContainer(max_number_faces, level, volume, midpoint, dx,
                                    num_faces, face_midpoints, face_areas, face_normals,
                                    face_connectivity, boundary_name, neighbor_faces)
        n_dims = length(midpoint)
        new{n_dims, eltype(midpoint), typeof(volume), max_number_faces,
            n_dims * max_number_faces}(level, volume, midpoint, dx, num_faces,
                                       face_midpoints, face_areas, face_normals,
                                       face_connectivity, boundary_name, neighbor_faces)
    end
end

@inline Base.ndims(::T8codeElementContainer{NDIMS}) where {NDIMS} = NDIMS
@inline function Base.eltype(::T8codeElementContainer{NDIMS, RealT, uEltype}) where {
                                                                                     NDIMS,
                                                                                     RealT,
                                                                                     uEltype
                                                                                     }
    uEltype
end

@inline is_ghost_cell(element, mesh) = element > ncells(mesh)

function Base.show(container::T8codeElementContainer)
    n_dims = length(container.midpoint)
    (; num_faces) = container
    println("level              = ", container.level)
    println("volume             = ", container.volume)
    println("midpoint           = ", container.midpoint)
    println("dx                 = ", container.dx)
    println("num_faces          = ", num_faces)
    println("face_midpoints     = ", container.face_midpoints[1:(n_dims * num_faces)])
    println("face_areas         = ", container.face_areas[1:num_faces])
    println("face_normals       = ", container.face_normals[1:(n_dims * num_faces)])
    println("face_connectivity  = ", container.face_connectivity[1:num_faces])
    println("neighbor_faces     = ", container.neighbor_faces[1:num_faces])
end

function init_fv_elements(mesh::T8codeMesh{2}, equations,
                          solver::FV, ::Type{uEltype}) where {uEltype}
    (; forest) = mesh
    # Check that the forest is a committed.
    @assert(t8_forest_is_committed(forest)==1)
    n_dims = ndims(mesh)
    (; max_number_faces) = mesh

    # Get the number of local elements of forest.
    num_local_elements = ncells(mesh)
    # Get the number of ghost elements of forest.
    num_ghost_elements = t8_forest_get_num_ghosts(forest)

    # Build an array of our data that is as long as the number of elements plus
    # the number of ghosts.
    elements = Array{T8codeElementContainer{n_dims, real(mesh), uEltype,
                                            max_number_faces,
                                            n_dims * max_number_faces}}(undef,
                                                                        num_local_elements +
                                                                        num_ghost_elements)

    init_fv_elements!(elements, mesh)

    # Exchange the neighboring data at MPI process boundaries.
    exchange_ghost_data(mesh, elements)

    return elements
end

function init_fv_elements!(elements, mesh::T8codeMesh)
    (; forest) = mesh
    n_dims = ndims(mesh)
    (; max_number_faces) = mesh

    midpoint = Vector{Cdouble}(undef, n_dims)

    face_midpoints = Matrix{Cdouble}(undef, 3, max_number_faces) # Need NDIMS=3 for t8code API. Also, consider that Julia is column major.
    face_areas = Vector{Cdouble}(undef, max_number_faces)
    face_normals = Matrix{Cdouble}(undef, 3, max_number_faces) # Need NDIMS=3 for t8code API. Also, consider that Julia is column major.
    face_connectivity = Vector{t8_locidx_t}(undef, max_number_faces)
    boundary_name = Vector{Symbol}(undef, max_number_faces)
    neighbor_faces = Vector{t8_locidx_t}(undef, max_number_faces)

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
            boundary_name .= Symbol("---")

            for iface in 1:num_faces
                # C++ is zero-indexed
                t8_forest_element_face_centroid(forest, itree, element, iface - 1,
                                                @views(face_midpoints[:, iface]))
                face_areas[iface] = t8_forest_element_face_area(forest, itree, element,
                                                                iface - 1)
                t8_forest_element_face_normal(forest, itree, element, iface - 1,
                                              @views(face_normals[:, iface]))

                # [ugly API, needs rework :/]
                neighids_ref = Ref{Ptr{t8_locidx_t}}()
                neighbors_ref = Ref{Ptr{Ptr{t8_element}}}()
                neigh_scheme_ref = Ref{Ptr{t8_eclass_scheme}}()

                dual_faces_ref = Ref{Ptr{Cint}}()
                num_neighbors_ref = Ref{Cint}()

                forest_is_balanced = Cint(1)

                t8_forest_leaf_face_neighbors(forest, itree, element,
                                              neighbors_ref, iface - 1, dual_faces_ref,
                                              num_neighbors_ref,
                                              neighids_ref, neigh_scheme_ref,
                                              forest_is_balanced)

                num_neighbors = num_neighbors_ref[]
                dual_faces = 1 .+ unsafe_wrap(Array, dual_faces_ref[], num_neighbors)
                neighids = 1 .+ unsafe_wrap(Array, neighids_ref[], num_neighbors)
                neighbors = unsafe_wrap(Array, neighbors_ref[], num_neighbors)
                neigh_scheme = neigh_scheme_ref[]

                if num_neighbors == 1
                    face_connectivity[iface] = neighids[1]
                    neighbor_faces[iface] = dual_faces[1]
                elseif num_neighbors > 1
                    error("Mortars are not supported yet.")
                elseif num_neighbors == 0 # No neighbors => boundary
                    boundary_name[iface] = mesh.boundary_names[iface, itree + 1]
                end

                # Free allocated memory.
                T8code.Libt8.sc_free(t8_get_package_id(), neighbors_ref[])
                T8code.Libt8.sc_free(t8_get_package_id(), dual_faces_ref[])
                T8code.Libt8.sc_free(t8_get_package_id(), neighids_ref[])
                # [/ugly API]
            end

            elements[current_index] = T8codeElementContainer(max_number_faces,
                                                             level,
                                                             volume,
                                                             Tuple(midpoint),
                                                             dx,
                                                             num_faces,
                                                             Tuple(@views(face_midpoints[1:n_dims,
                                                                                         :])),
                                                             Tuple(face_areas),
                                                             Tuple(@views(face_normals[1:n_dims,
                                                                                       :])),
                                                             Tuple(face_connectivity),
                                                             Tuple(boundary_name),
                                                             Tuple(neighbor_faces))
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
function init_fv_interfaces(mesh::T8codeMesh, equations,
                            solver::FV, elements, uEltype)
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

    # I tried it to do it like for the existing T8codeMesh routines with
    # init_interfaces!(interfaces, mesh)
    # The problem was that I need the face id of both elements for every interface.
    # That is not needed for DG code since it is handled with the indices there.

    init_fv_interfaces!(interfaces, mesh, elements)

    return interfaces
end

function init_fv_interfaces!(interfaces, mesh::T8codeMesh, elements)
    # Note: In t8code, the routine 't8code_forest_iterate' is not implemented yet.

    idx = 1
    for element in 1:ncells(mesh)
        (; face_connectivity, num_faces, face_midpoints, neighbor_faces) = elements[element]
        for (face, neighbor) in enumerate(face_connectivity[1:num_faces])
            if neighbor < element
                continue
            end

            # face_midpoint = Trixi.get_variable_wrapped(face_midpoints, equations, face)
            face_neighbor = neighbor_faces[face]
            # face_midpoint_neighbor = Trixi.get_variable_wrapped(elements[neighbor].face_midpoints,
            #                                                     equations,
            #                                                     face_neighbor)
            interfaces.neighbor_ids[1, idx] = element
            interfaces.neighbor_ids[2, idx] = neighbor

            interfaces.faces[1, idx] = face
            interfaces.faces[2, idx] = face_neighbor

            idx += 1
        end
    end

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
function init_fv_boundaries(mesh::T8codeMesh, equations, solver::FV, elements, uEltype)
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

    if n_boundaries > 0
        # See above
        # init_boundaries!(boundaries, mesh)

        init_fv_boundaries!(boundaries, mesh, elements)
    end

    return boundaries
end

function init_fv_boundaries!(boundaries, mesh::T8codeMesh, elements)
    # Note: In t8code, the routine 't8code_forest_iterate' is not implemented yet.

    idx = 1
    for element in 1:ncells(mesh)
        (; face_connectivity, num_faces, boundary_name) = elements[element]
        for (face, neighbor) in enumerate(face_connectivity[1:num_faces])
            if neighbor > 0
                continue
            end
            boundaries.neighbor_ids[idx] = element
            boundaries.faces[idx] = face

            boundaries.name[idx] = boundary_name[face]

            idx += 1
        end
    end

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
