# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

# The uniform refinement level of the forest.
const refinement_level = 4

# The data that we want to store for each element.
struct T8codeElementContainer{NDIMS, MAX_NUMBER_FACES, NDIMS_MAX_NUMBER_FACES}
    level    :: Cint
    volume   :: Cdouble
    midpoint :: NTuple{NDIMS, Cdouble}
    dx       :: Cdouble # Characteristic length (for CFL condition).

    num_faces         :: Cint
    face_midpoints    :: NTuple{NDIMS_MAX_NUMBER_FACES, Cdouble}
    face_areas        :: NTuple{MAX_NUMBER_FACES, Cdouble}
    face_normals      :: NTuple{NDIMS_MAX_NUMBER_FACES, Cdouble}
    face_connectivity :: NTuple{MAX_NUMBER_FACES, t8_locidx_t} # ids of the face neighbors
    neighbor_faces    :: NTuple{MAX_NUMBER_FACES, t8_locidx_t}

    function T8codeElementContainer(max_number_faces, level, volume, midpoint, dx,
                                    num_faces, face_midpoints, face_areas, face_normals,
                                    face_connectivity, neighbor_faces)
        n_dims = length(midpoint)
        new{n_dims, max_number_faces, n_dims * max_number_faces}(level, volume,
                                                                 midpoint, dx,
                                                                 num_faces,
                                                                 face_midpoints,
                                                                 face_areas,
                                                                 face_normals,
                                                                 face_connectivity,
                                                                 neighbor_faces)
    end
end

function Base.show(container::T8codeElementContainer)
    n_dims = length(container.midpoint)
    @unpack num_faces = container
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

function init_elements(mesh::T8codeFVMesh, RealT, uEltype)
    # Initialize container
    elements = init_elements(mesh)

    # Exchange the neighboring data at MPI process boundaries.
    exchange_ghost_data(mesh, elements)

    return elements
end

function init_elements(mesh::T8codeFVMesh)
    @unpack forest = mesh
    # Check that the forest is a committed.
    @assert(t8_forest_is_committed(forest)==1)
    n_dims = ndims(mesh)
    @unpack max_number_faces = mesh

    # Get the number of local elements of forest.
    num_local_elements = mesh.number_elements
    # Get the number of ghost elements of forest.
    num_ghost_elements = t8_forest_get_num_ghosts(forest)

    # Build an array of our data that is as long as the number of elements plus
    # the number of ghosts.
    elements = Array{
                     T8codeElementContainer{n_dims, max_number_faces,
                                            n_dims * max_number_faces}}(undef,
                                                                        num_local_elements +
                                                                        num_ghost_elements)

    midpoint = Vector{Cdouble}(undef, n_dims)

    face_midpoints = Matrix{Cdouble}(undef, 3, max_number_faces) # Need NDIMS=3 for t8code API. Also, consider that Julia is column major.
    face_areas = Vector{Cdouble}(undef, max_number_faces)
    face_normals = Matrix{Cdouble}(undef, 3, max_number_faces) # Need NDIMS=3 for t8code API. Also, consider that Julia is column major.
    face_connectivity = Vector{t8_locidx_t}(undef, max_number_faces)
    neighbor_faces = Vector{t8_locidx_t}(undef, max_number_faces)

    # Loop over all local trees in the forest.
    current_index = 0
    for itree in 0:(mesh.number_trees_local - 1)
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

            for iface in 1:num_faces
                t8_forest_element_face_centroid(forest, itree, element, iface - 1,
                                                @views(face_midpoints[:, iface]))
                face_areas[iface] = t8_forest_element_face_area(forest, itree, element,
                                                                iface - 1) # C++ is zero-indexed
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

                face_connectivity[iface] = neighids[1]
                neighbor_faces[iface] = dual_faces[1]

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
                                                             Tuple(neighbor_faces))
        end
    end

    return elements
end

mutable struct T8codeInterfaceContainer{NDIMS, uEltype <: Real} <: AbstractContainer
    u::Array{uEltype, 3}                # [primary/secondary, variable, interface]
    neighbor_ids::Matrix{Int}           # [primary/secondary, interface]
    faces::Matrix{uEltype}              # [primary/secondary, interface]

    # internal `resize!`able storage
    _u::Vector{uEltype}
    _neighbor_ids::Vector{Int}
    _faces::Vector{uEltype}
end

@inline ninterfaces(solver::FV, cache) = ninterfaces(cache.interfaces)
@inline function ninterfaces(interfaces::T8codeInterfaceContainer)
    size(interfaces.neighbor_ids, 2)
end
@inline Base.ndims(::T8codeInterfaceContainer{NDIMS}) where {NDIMS} = NDIMS

# See explanation of Base.resize! for the element container
function Base.resize!(interfaces::T8codeInterfaceContainer, capacity)
    @unpack _u, _neighbor_ids, _faces = interfaces

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
function init_interfaces(mesh::T8codeFVMesh, equations, elements)
    NDIMS = ndims(mesh)
    uEltype = eltype(elements[1].volume)

    # Initialize container # TODO: nonperiodic
    n_interfaces = zero(Int)
    for element in 1:(mesh.number_elements)
        for face in 1:(elements[element].num_faces)
            if elements[element].face_connectivity[face] <= mesh.number_elements
                n_interfaces += 0.5
            elseif elements[element].face_connectivity[face] <= length(elements)
                n_interfaces += 1.0
            else
                error("Should not occur for periodic domains.")
            end
        end
    end
    @assert Int(n_interfaces)==n_interfaces "Something is wrong!"
    n_interfaces = Int(n_interfaces)

    _u = Vector{uEltype}(undef,
                         2 * nvariables(equations) * n_interfaces)
    u = unsafe_wrap(Array, pointer(_u),
                    (2, nvariables(equations), n_interfaces))

    _neighbor_ids = Vector{Int}(undef, 2 * n_interfaces)
    neighbor_ids = unsafe_wrap(Array, pointer(_neighbor_ids), (2, n_interfaces))

    _faces = Vector{Int}(undef, 2 * n_interfaces)
    faces = unsafe_wrap(Array, pointer(_faces), (2, n_interfaces))

    interfaces = T8codeInterfaceContainer{NDIMS, uEltype}(u, neighbor_ids, faces, _u,
                                                          _neighbor_ids, _faces)

    init_interfaces!(interfaces, mesh, equations, elements)

    return interfaces
end

function init_interfaces!(interfaces, mesh::T8codeFVMesh, equations, elements)
    # Note: In t8code, the routine 't8code_forest_iterate' is not implemented yet.

    idx = 1
    for element in 1:(mesh.number_elements)
        (; face_connectivity, num_faces, face_midpoints, neighbor_faces) = elements[element]
        for (face, neighbor) in enumerate(face_connectivity[1:num_faces])
            if neighbor < element
                continue
            end

            face_midpoint = Trixi.get_variable_wrapped(face_midpoints, equations, face)
            face_neighbor = neighbor_faces[face]
            face_midpoint_neighbor = Trixi.get_variable_wrapped(elements[neighbor].face_midpoints,
                                                                equations,
                                                                face_neighbor)
            interfaces.neighbor_ids[1, idx] = element
            interfaces.neighbor_ids[2, idx] = neighbor

            interfaces.faces[1, idx] = face
            interfaces.faces[2, idx] = face_neighbor

            idx += 1
        end
    end

    return interfaces
end

@inline eachinterface(solver::FV, cache) = Base.OneTo(ninterfaces(solver, cache))

function init_solution!(mesh::T8codeFVMesh, equations)
    @unpack forest = mesh
    # Check that the forest is a committed.
    @assert(t8_forest_is_committed(forest)==1)

    # Get the number of local elements of forest.
    num_local_elements = mesh.number_elements
    # Get the number of ghost elements of forest.
    num_ghost_elements = t8_forest_get_num_ghosts(forest)

    # Build an array of our data that is as long as the number of elements plus
    # the number of ghosts.
    u_ = Vector{
                T8codeSolutionContainer{nvariables(equations)}}(undef,
                                                                num_local_elements +
                                                                num_ghost_elements)

    return u_
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
    @unpack u_ = cache
    for element in eachelement(mesh, solver)
        u_[element] = T8codeSolutionContainer(Tuple(get_node_vars(u, equations, solver,
                                                                  element)))
    end
    exchange_ghost_data(mesh, u_)

    return nothing
end
end # @muladd
