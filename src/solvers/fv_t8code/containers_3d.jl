# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

@inline function init_mortar_neighbor_ids!(mortars::T8codeFVMortarContainer{3},
                                           my_index, other_indices, mortar_id)
    # Here, the large element is local on the current rank. Therefore, we have information about all small ones.
    mortars.neighbor_ids[end, mortar_id] = my_index

    mortars.neighbor_ids[1, mortar_id] = other_indices[1]
    mortars.neighbor_ids[2, mortar_id] = other_indices[2]
    mortars.neighbor_ids[3, mortar_id] = other_indices[3]
    mortars.neighbor_ids[4, mortar_id] = other_indices[4]

    # Set `n_local_elements_small[mortar_id]` to the maximum = 2^(ndims - 1) = 4.
    mortars.n_local_elements_small[mortar_id] = 4

    return nothing
end

@inline function init_mortar_neighbor_ids_first!(mortars::T8codeFVMortarContainer{3},
                                                 my_index, other_index, mortar_id)
    # Here, the large element is a ghost element.
    # The element `my_index` is local on the current rank and is the first one at this mortar.
    # Only add the information about the large element and this first one.
    mortars.neighbor_ids[end, mortar_id] = other_index

    mortars.neighbor_ids[1, mortar_id] = my_index
    mortars.neighbor_ids[2, mortar_id] = -1
    mortars.neighbor_ids[3, mortar_id] = -1
    mortars.neighbor_ids[4, mortar_id] = -1

    mortars.n_local_elements_small[mortar_id] = 1

    return nothing
end

@inline function init_mortar_neighbor_ids_fill!(mortars::T8codeFVMortarContainer{3},
                                                my_index, other_index, mortar_id)
    # Here, the large element is a ghost element.
    # The element `my_index` is local on the current rank, but there were already element(s) before at this mortar.
    # Check the information of the large element and add the new information.
    @assert mortars.neighbor_ids[end, mortar_id] == other_index

    elements_added_to_mortar = mortars.n_local_elements_small[mortar_id]
    @assert mortars.neighbor_ids[elements_added_to_mortar + 1, mortar_id] == -1
    mortars.neighbor_ids[elements_added_to_mortar + 1, mortar_id] = my_index

    mortars.n_local_elements_small[mortar_id] += 1

    return nothing
end

@inline function init_mortar_faces!(mortars::T8codeFVMortarContainer{3},
                                    my_face, other_faces, mortar_id)
    mortars.faces[end, mortar_id] = my_face

    mortars.faces[1, mortar_id] = other_faces[1]
    mortars.faces[2, mortar_id] = other_faces[2]
    mortars.faces[3, mortar_id] = other_faces[3]
    mortars.faces[4, mortar_id] = other_faces[4]

    return nothing
end

@inline function init_mortar_faces_first!(mortars::T8codeFVMortarContainer{3},
                                          my_face, other_face, mortar_id)
    mortars.faces[end, mortar_id] = other_face

    mortars.faces[1, mortar_id] = my_face
    mortars.faces[2, mortar_id] = -1
    mortars.faces[3, mortar_id] = -1
    mortars.faces[4, mortar_id] = -1

    return nothing
end

@inline function init_mortar_faces_fill!(mortars::T8codeFVMortarContainer{3},
                                         my_face, other_face, mortar_id)
    @assert mortars.faces[end, mortar_id] == other_face

    elements_added_to_mortar = mortars.n_local_elements_small[mortar_id]
    @assert mortars.faces[elements_added_to_mortar, mortar_id] == -1
    mortars.faces[elements_added_to_mortar, mortar_id] = my_face

    return nothing
end
end # @muladd
