# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

# The data that we want to store for each element.
mutable struct T8codeElementData{NDIMS, MAX_NUM_FACES, NDIMS_MAX_NUM_FACES, T8_LOCIDX_T}
    level             :: Cint
    volume            :: Cdouble
    midpoint          :: NTuple{NDIMS, Cdouble}
    dx                :: Cdouble # Characteristic length (for CFL condition).

    num_faces         :: Cint
    face_areas        :: NTuple{MAX_NUM_FACES, Cdouble}
    face_normals      :: NTuple{NDIMS_MAX_NUM_FACES, Cdouble}
    face_connectivity :: NTuple{MAX_NUM_FACES, T8_LOCIDX_T} # ids of the face neighbors

    function T8codeElementData(max_num_faces, level, volume, midpoint, dx, num_faces, face_areas,
                               face_normals, face_connectivity)
        n_dims = length(midpoint)
        new{n_dims, max_num_faces, n_dims * max_num_faces,
            eltype(face_connectivity)}(level, volume, midpoint, dx, num_faces, face_areas,
                                       face_normals, face_connectivity)
    end
end

# function T8codeElementData(mesh, RealT, uEltype, t8_locidx_t)
#     n_dims = ndims(mesh)
#     @unpack max_number_faces = mesh

#     level = typemin(Cint)
#     volume = typemin(Cdouble)
#     dx = typemin(Cdouble)
#     num_faces = typemin(Cint)

#     midpoint = Vector{Cdouble}(undef, n_dims)

#     face_areas = Vector{Cdouble}(undef, max_number_faces)
#     face_normals = Matrix{Cdouble}(undef, 3, max_number_faces) # Need NDIMS=3 for t8code API. Also, consider that Julia is column major.
#     face_connectivity = Vector{t8_locidx_t}(undef, max_number_faces)

#     T8codeElementData{n_dims, max_number_faces,
#                       3 * max_number_faces,
#                       t8_locidx_t}(level, volume,
#                                    Tuple(midpoint), #= Ich kann hier kein Tuple nutzen, da sonst pointer(midpoint) nichts funktioniert.=#dx,
#                                    num_faces, face_areas, # kein Tuple möglich wegen setindex! (?)
#                                    face_normals[1:3,:], face_connectivity, # kein Tuple möglich bei .= -1, und getindex()
#                                    )
# end

@inline Base.ndims(::T8codeElementData{NDIMS}) where {NDIMS} = NDIMS


# Container data structure (structure-of-arrays style) for DG elements
mutable struct T8codeElementContainer{ElementData} <: AbstractContainer
    element_data::Vector{ElementData} # [element_idx]

    function T8codeElementContainer(element_data)
        new{eltype(element_data)}(element_data)
    end
end

# function T8codeElementContainer(mesh, NDIMS, RealT, uEltype, T8_LOCIDX_T)
#     # Get the number of local and ghost elements of forest.
#     num_local_elements = t8_forest_get_local_num_elements(mesh.forest)
#     num_ghost_elements = t8_forest_get_num_ghosts(mesh.forest)
#     n_elements = num_local_elements + num_ghost_elements

#     # data = T8codeElementData(mesh, RealT, uEltype, T8_LOCIDX_T)
#     element_data = Vector{T8codeElementData}(undef, n_elements)
#     # for i in 1:n_elements
#     #     element_data[i] = data
#     # end

#     T8codeElementContainer{T8codeElementData}(element_data)
# end

Base.eltype(elements::T8codeElementContainer) = eltype(elements.element_data[1].volume)

@inline nelements(elements::T8codeElementContainer) = length(elements.element_data) # TODO: Das scheinen nicht alle Elemente zu sein...
@inline eachelement(elements::T8codeElementContainer) = Base.OneTo(nelements(elements))

function Base.resize!(elements::T8codeElementContainer, capacity)
    resize!(elements.element_data, capacity)

    return nothing
end

function init_elements(mesh::T8codeMesh{NDIMS}, RealT, uEltype) where NDIMS
    # Initialize container
    element_data = init_element_data(mesh)

    # Exchange the neighboring data at MPI process boundaries.
    exchange_ghost_data(mesh, element_data)

    elements = T8codeElementContainer(element_data)
end

function init_element_data(mesh)
    @unpack forest = mesh
    # @unpack element_data = elements
    # Check that the forest is a committed.
    @assert(t8_forest_is_committed(forest) == 1)
    n_dims = ndims(mesh)
    @unpack max_number_faces = mesh

    # Get the number of local elements of forest.
    num_local_elements = t8_forest_get_local_num_elements(forest)
    # Get the number of ghost elements of forest.
    num_ghost_elements = t8_forest_get_num_ghosts(forest)

    # Build an array of our data that is as long as the number of elements plus
    # the number of ghosts.
    element_data = Array{T8codeElementData}(undef, num_local_elements + num_ghost_elements)

    # Get the number of trees that have elements of this process.
    num_local_trees = t8_forest_get_num_local_trees(forest)

    midpoint = Vector{Cdouble}(undef,n_dims)

    face_areas = Vector{Cdouble}(undef,max_number_faces)
    face_normals = Matrix{Cdouble}(undef,3,max_number_faces) # Need NDIMS=3 for t8code API. Also, consider that Julia is column major.
    face_connectivity = Vector{t8_locidx_t}(undef,max_number_faces)

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
                # dual_faces    = 1 .+ unsafe_wrap(Array, dual_faces_ref[], num_neighbors)
                neighids      = 1 .+ unsafe_wrap(Array, neighids_ref[], num_neighbors)
                # neighbors     = unsafe_wrap(Array, neighbors_ref[], num_neighbors)
                # neigh_scheme  = neigh_scheme_ref[]

                face_connectivity[iface] = neighids[1]

                # Free allocated memory.
                sc_free(t8_get_package_id(), neighbors_ref[])
                sc_free(t8_get_package_id(), dual_faces_ref[])
                sc_free(t8_get_package_id(), neighids_ref[])
                # [/ugly API]
            end

            element_data[current_index] = T8codeElementData(max_number_faces,
                level,
                volume,
                Tuple(midpoint),
                dx,
                num_faces,
                Tuple(face_areas),
                Tuple(@views(face_normals[1:2,:])),
                Tuple(face_connectivity)
            )
        end
    end

    return element_data
end


# Each process has computed the data entries for its local elements. In order
# to get the values for the ghost elements, we use
# t8_forest_ghost_exchange_data. Calling this function will fill all the ghost
# entries of our element data array with the value on the process that owns the
# corresponding element.
function exchange_ghost_data(mesh, element_data::Vector{T8codeElementData})
    # t8_forest_ghost_exchange_data expects an sc_array (of length num_local_elements + num_ghosts).
    # We wrap our data array to an sc_array.
    sc_array_wrapper = sc_array_new_data(pointer(element_data), sizeof(element_data[1]), length(element_data))

    # Carry out the data exchange. The entries with indices > num_local_elements will get overwritten.
    t8_forest_ghost_exchange_data(mesh.forest, sc_array_wrapper)

    # Destroy the wrapper array. This will not free the data memory since we used sc_array_new_data.
    sc_array_destroy(sc_array_wrapper)
end

# # Container data structure (structure-of-arrays style) for DG elements
# mutable struct T8codeElementContainer{NDIMS, RealT <: Real, uEltype <: Real, T8_LOCIDX_T}
#     <: AbstractContainer
#     level::Vector{Int}          # [element]
#     volume::Vector{RealT}       # [element]
#     midpoint::Array{RealT, 2}   # [index, element]
#     dx::Vector{RealT}           # [element] # Characteristic length (for CFL condition).

#     num_faces::Vector{Int}                  # [element]
#     face_areas::Array{RealT, 2}             # [face, element]
#     face_normals::Array{RealT, 3}           # [index, face, element]
#     face_connectivity::Array{T8_LOCIDX_T, 2}# [face, element] # ids of the face neighbors

#     # internal `resize!`able storage
#     _midpoint::Vector{RealT}
#     _face_areas::Vector{RealT}
#     _face_normals::Vector{RealT}
#     _face_connectivity::Vector{RealT}
# end

# Base.eltype(elements::T8codeElementContainer) = eltype(elements.surface_flux_values)

# function Base.resize!(elements::T8codeElementContainer, capacity)
#     n_dims = size(elements.midpoint, 1)
#     max_number_faces = size(elements.face_areas, 1)

#     @unpack _midpoint, _face_areas, _face_normals = elements

#     resize!(level, capacity)
#     resize!(volume, capacity)

#     resize!(_midpoint, n_dims * capacity)
#     elements.midpoint = unsafe_wrap(Array, pointer(_midpoint), (ndims, capacity))

#     resize!(dx, capacity)
#     resize!(num_faces, capacity)

#     resize!(_face_areas, max_number_faces * capacity)
#     elements.face_areas = unsafe_wrap(Array, pointer(_face_areas), (max_number_faces, capacity))

#     resize!(_face_normals, 3#= Need NDIMS=3 for t8code API. =# * max_number_faces * capacity)
#     elements.face_normals = unsafe_wrap(Array, pointer(_face_normals), (3, max_number_faces, capacity))

#     resize!(_face_connectivity, max_number_faces * capacity)
#     elements.face_connectivity = unsafe_wrap(Array, pointer(_face_connectivity), (max_number_faces, capacity))

#     return nothing
# end

# function T8codeElementContainer{NDIMS, RealT, uEltype, T8_LOCIDX_T}(capacity::Integer,
#                                                        max_number_faces::Integer) where {NDIMS,
#                                                                                          RealT <: Real,
#                                                                                          uEltype <: Real,
#                                                                                          T8_LOCIDX_T}
#     nan_RealT = convert(RealT, NaN)
#     nan_uEltype = convert(uEltype, NaN)

#     # Initialize fields with defaults
#     level = fill(typemin(Int), capacity)
#     volume = fill(nan_RealT, capacity)

#     _midpoint = fill(nan_RealT, 3#= Need NDIMS=3 for t8code API=# * capacity)
#     midpoint = unsafe_wrap(Array, pointer(_midpoint), (3, capacity))

#     dx = fill(nan_RealT, capacity)
#     num_faces = fill(typemin(Int), capacity)

#     _face_areas = fill(nan_RealT, max_number_faces * capacity)
#     face_areas = unsafe_wrap(Array, pointer(_face_areas), (max_number_faces, capacity))

#     _face_normals = fill(nan_RealT, NDIMS * max_number_faces * capacity)
#     face_normals = unsafe_wrap(Array, pointer(_face_normals),
#                                (NDIMS, max_number_faces, capacity))

#     _face_connectivity = fill(nan_RealT, max_number_faces * capacity)
#     face_connectivity = unsafe_wrap(Array, pointer(_face_connectivity), (max_number_faces, capacity))

#     return T8codeElementContainer{NDIMS, RealT, uEltype,
#                                   T8_LOCIDX_T}(level, volume, midpoint,
#                                                  dx, num_faces, face_areas,
#                                                  face_normals, face_connectivity,
#                                                  _midpoint, _face_areas,
#                                                  _face_normals, _face_connectivity)
# end
end # @muladd

