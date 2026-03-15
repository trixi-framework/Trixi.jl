@muladd begin
#! format: noindent

# Use for `T8CodeMesh` the Gauss-Lobatto-Legendre (GLL) hard-coded variant,
# i.e., take interface normals from the outer volume nodes.
function calc_interface_flux!(surface_flux_values,
                              mesh::T8codeMesh{2},
                              have_nonconservative_terms,
                              equations, surface_integral, dg::DG, cache)
    @unpack neighbor_ids, node_indices = cache.interfaces
    @unpack contravariant_vectors = cache.elements
    index_range = eachnode(dg)
    index_end = last(index_range)

    @threaded for interface in eachinterface(dg, cache)
        # Get element and side index information on the primary element
        primary_element = neighbor_ids[1, interface]
        primary_indices = node_indices[1, interface]
        primary_direction = indices2direction(primary_indices)

        # Create the local i,j indexing on the primary element used to pull normal direction information
        i_primary_start, i_primary_step = index_to_start_step_2d(primary_indices[1],
                                                                 index_range)
        j_primary_start, j_primary_step = index_to_start_step_2d(primary_indices[2],
                                                                 index_range)

        i_primary = i_primary_start
        j_primary = j_primary_start

        # Get element and side index information on the secondary element
        secondary_element = neighbor_ids[2, interface]
        secondary_indices = node_indices[2, interface]
        secondary_direction = indices2direction(secondary_indices)

        # Initiate the secondary index to be used in the surface for loop.
        # This index on the primary side will always run forward but
        # the secondary index might need to run backwards for flipped sides.
        if :i_backward in secondary_indices
            node_secondary = index_end
            node_secondary_step = -1
        else
            node_secondary = 1
            node_secondary_step = 1
        end

        for node in eachnode(dg)
            # Get the normal direction on the primary element.
            # Contravariant vectors at interfaces in negative coordinate direction
            # are pointing inwards. This is handled by `get_normal_direction`.
            normal_direction = get_normal_direction(primary_direction,
                                                    contravariant_vectors,
                                                    i_primary, j_primary,
                                                    primary_element)

            calc_interface_flux!(surface_flux_values, mesh, have_nonconservative_terms,
                                 equations,
                                 surface_integral, dg, cache,
                                 interface, normal_direction,
                                 node, primary_direction, primary_element,
                                 node_secondary, secondary_direction, secondary_element)

            # Increment primary element indices to pull the normal direction
            i_primary += i_primary_step
            j_primary += j_primary_step
            # Increment the surface node index along the secondary element
            node_secondary += node_secondary_step
        end
    end

    return nothing
end
end # @muladd
