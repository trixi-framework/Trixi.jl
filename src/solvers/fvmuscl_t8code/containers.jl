# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

const NDIMS = 2
const MAX_NUM_FACES = 4

# The uniform refinement level of the forest.
const refinement_level = 4

# The data that we want to store for each element.
struct data_per_element_t
    level             :: Cint
    volume            :: Cdouble
    midpoint          :: NTuple{NDIMS,Cdouble}
    dx                :: Cdouble # Characteristic length (for CFL condition).
  
    num_faces         :: Cint
    face_areas        :: NTuple{MAX_NUM_FACES,Cdouble}
    face_normals      :: NTuple{MAX_NUM_FACES*NDIMS,Cdouble}
    face_connectivity :: NTuple{MAX_NUM_FACES,t8_locidx_t} # ids of the face neighbors
  end
  
function Base.show(data::data_per_element_t)
    println("level              = ", data.level)
    println("volume             = ", data.volume)
    println("midpoint           = ", data.midpoint)
    println("dx                 = ", data.dx)
    println("num_faces          = ", data.num_faces)
    println("face_areas         = ", data.face_areas)
    println("face_normals       = ", data.face_normals)
    println("face_connectivity  = ", data.face_connectivity)  
end

function init_elements(mesh::T8codeMesh{NDIMS}, RealT, uEltype) where NDIMS
    # Initialize container
    elements = init_elements(mesh)

    # Exchange the neighboring data at MPI process boundaries.
    exchange_ghost_data(mesh, elements)

    return elements
end

function init_elements(mesh::T8codeMesh)
    @unpack forest = mesh
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
    elements = Array{data_per_element_t}(undef, num_local_elements + num_ghost_elements)

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
                dual_faces    = 1 .+ unsafe_wrap(Array, dual_faces_ref[], num_neighbors)
                neighids      = 1 .+ unsafe_wrap(Array, neighids_ref[], num_neighbors)
                neighbors     = unsafe_wrap(Array, neighbors_ref[], num_neighbors)
                neigh_scheme  = neigh_scheme_ref[]

                face_connectivity[iface] = neighids[1]
 
                # Free allocated memory.
                T8code.Libt8.sc_free(t8_get_package_id(), neighbors_ref[])
                T8code.Libt8.sc_free(t8_get_package_id(), dual_faces_ref[])
                T8code.Libt8.sc_free(t8_get_package_id(), neighids_ref[])
                # [/ugly API]
            end


        elements[current_index] = data_per_element_t(
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

    return elements
end


# Each process has computed the data entries for its local elements. In order
# to get the values for the ghost elements, we use
# t8_forest_ghost_exchange_data. Calling this function will fill all the ghost
# entries of our element data array with the value on the process that owns the
# corresponding element.
function exchange_ghost_data(mesh, elements)
    # t8_forest_ghost_exchange_data expects an sc_array (of length num_local_elements + num_ghosts).
    # We wrap our data array to an sc_array.
    sc_array_wrapper = T8code.Libt8.sc_array_new_data(pointer(elements), sizeof(data_per_element_t), length(elements))

    # Carry out the data exchange. The entries with indices > num_local_elements will get overwritten.
    t8_forest_ghost_exchange_data(mesh.forest, sc_array_wrapper)

    # Destroy the wrapper array. This will not free the data memory since we used sc_array_new_data.
    T8code.Libt8.sc_array_destroy(sc_array_wrapper)
end
end # @muladd
