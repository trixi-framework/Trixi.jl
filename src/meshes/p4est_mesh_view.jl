# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

"""
    P4estMeshView{NDIMS, NDIMS_AMBIENT, RealT <: Real, Parent} <: AbstractMesh{NDIMS}

A view on a [`P4estMesh`](@ref).
"""
mutable struct P4estMeshView{NDIMS, NDIMS_AMBIENT, RealT <: Real, Parent} <:
               AbstractMesh{NDIMS}
    const parent::Parent
    const cell_ids::Vector{Int}
    unsaved_changes::Bool
    current_filename::String
end

"""
    P4estMeshView(parent; cell_ids)

Create a `P4estMeshView` on a [`P4estMesh`](@ref) parent.

# Arguments
- `parent`: the parent `P4estMesh`.
- `cell_ids`: array of cell ids that are part of this view.
"""
function P4estMeshView(parent::P4estMesh{NDIMS, NDIMS_AMBIENT, RealT},
                       cell_ids::Vector) where {NDIMS, NDIMS_AMBIENT, RealT}
    return P4estMeshView{NDIMS, NDIMS_AMBIENT, RealT, typeof(parent)}(parent, cell_ids,
                                                                      parent.unsaved_changes,
                                                                      parent.current_filename)
end

@inline Base.ndims(::P4estMeshView{NDIMS}) where {NDIMS} = NDIMS
@inline Base.real(::P4estMeshView{NDIMS, NDIMS_AMBIENT, RealT}) where {NDIMS, NDIMS_AMBIENT, RealT} = RealT

@inline ncells(mesh::P4estMeshView) = length(mesh.cell_ids)

function extract_p4est_mesh_view(elements_parent,
                                 interfaces_parent,
                                 boundaries_parent,
                                 mortars_parent,
                                 mesh,
                                 equations,
                                 dg,
                                 ::Type{uEltype}) where {uEltype <: Real}
    # Create deepcopy to get completely independent elements container
    elements = deepcopy(elements_parent)
    resize!(elements, length(mesh.cell_ids))

    # Copy relevant entries from parent mesh
    @views elements.inverse_jacobian .= elements_parent.inverse_jacobian[..,
                                                                         mesh.cell_ids]
    @views elements.jacobian_matrix .= elements_parent.jacobian_matrix[..,
                                                                       mesh.cell_ids]
    @views elements.node_coordinates .= elements_parent.node_coordinates[..,
                                                                         mesh.cell_ids]
    @views elements.contravariant_vectors .= elements_parent.contravariant_vectors[..,
                                                                                   mesh.cell_ids]
    @views elements.surface_flux_values .= elements_parent.surface_flux_values[..,
                                                                               mesh.cell_ids]
    # Extract interfaces that belong to mesh view.
    interfaces = extract_interfaces(mesh, interfaces_parent)

    # Extract boundaries of this mesh view.
    boundaries = extract_boundaries(mesh, boundaries_parent, interfaces_parent,
                                    interfaces)

    # Extract mortars that are entirely within this mesh view.
    mortars = extract_mortars(mesh, mortars_parent)

    # Get the global elements ids of the neighbors.
    neighbor_ids_global = extract_neighbor_ids_global(mesh, boundaries_parent,
                                                      interfaces_parent,
                                                      boundaries, elements_parent)

    return elements, interfaces, boundaries, mortars, neighbor_ids_global
end

# Remove all interfaces that have a tuple of neighbor_ids where at least one is
# not part of this mesh view, i.e. mesh.cell_ids, and return the new interface container.
function extract_interfaces(mesh::P4estMeshView, interfaces_parent)
    # Identify interfaces that need to be retained
    mask = BitArray(undef, ninterfaces(interfaces_parent))
    # Loop over all interfaces (index 2).
    for interface in 1:size(interfaces_parent.neighbor_ids)[2]
        mask[interface] = (interfaces_parent.neighbor_ids[1, interface] in mesh.cell_ids) &&
                          (interfaces_parent.neighbor_ids[2, interface] in mesh.cell_ids)
    end

    # Create deepcopy to get completely independent interfaces container
    interfaces = deepcopy(interfaces_parent)
    resize!(interfaces, sum(mask))

    # Copy relevant entries from parent mesh
    @views interfaces.u .= interfaces_parent.u[.., mask]
    @views interfaces.node_indices .= interfaces_parent.node_indices[.., mask]
    @views neighbor_ids = interfaces_parent.neighbor_ids[.., mask]

    # Transform the global (parent) indices into local (view) indices.
    interfaces.neighbor_ids = zeros(Int, size(neighbor_ids))
    for interface in 1:size(neighbor_ids)[2]
        interfaces.neighbor_ids[1, interface] = findall(id -> id ==
                                                              neighbor_ids[1, interface],
                                                        mesh.cell_ids)[1]
        interfaces.neighbor_ids[2, interface] = findall(id -> id ==
                                                              neighbor_ids[2, interface],
                                                        mesh.cell_ids)[1]
    end

    return interfaces
end

# Remove all boundaries that are not part of this p4est mesh view and add new boundaries
# that were interfaces of the parent mesh.
function extract_boundaries(mesh::P4estMeshView,
                            boundaries_parent, interfaces_parent,
                            interfaces)
    # Remove all boundaries that are not part of this p4est mesh view.
    boundaries = deepcopy(boundaries_parent)
    mask = BitArray(undef, nboundaries(boundaries_parent))
    for boundary in 1:size(boundaries_parent.neighbor_ids)[1]
        mask[boundary] = boundaries_parent.neighbor_ids[boundary] in mesh.cell_ids
    end
    boundaries.neighbor_ids = global_element_id_to_local(boundaries_parent.neighbor_ids[mask],
                                                         mesh)
    boundaries.name = boundaries_parent.name[mask]
    boundaries.node_indices = boundaries_parent.node_indices[mask]

    # Add new boundaries that were interfaces of the parent mesh.
    # Loop over all interfaces (index 2).
    for interface in 1:size(interfaces_parent.neighbor_ids)[2]
        if ((interfaces_parent.neighbor_ids[1, interface] in mesh.cell_ids) ⊻
            (interfaces_parent.neighbor_ids[2, interface] in mesh.cell_ids))
            # Determine which of the ids is part of the mesh view.
            if interfaces_parent.neighbor_ids[1, interface] in mesh.cell_ids
                neighbor_id = interfaces_parent.neighbor_ids[1, interface]
                view_idx = 1
            else
                neighbor_id = interfaces_parent.neighbor_ids[2, interface]
                view_idx = 2
            end

            # Update the neighbor ids.
            push!(boundaries.neighbor_ids,
                  global_element_id_to_local(neighbor_id, mesh))
            # Update the boundary names.
            if interfaces_parent.node_indices[view_idx, interface] ==
               (:end, :i_forward)
                push!(boundaries.name, :x_pos)
            elseif interfaces_parent.node_indices[view_idx, interface] ==
                   (:begin, :i_forward)
                push!(boundaries.name, :x_neg)
            elseif interfaces_parent.node_indices[view_idx, interface] ==
                   (:i_forward, :end)
                push!(boundaries.name, :y_pos)
            else
                push!(boundaries.name, :y_neg)
            end

            # Update the node indices.
            push!(boundaries.node_indices,
                  interfaces_parent.node_indices[view_idx, interface])
        end
    end

    # Create the boundary vector for u, which will be populated later.
    boundaries.u = zeros(typeof(boundaries_parent.u).parameters[1],
                         (size(boundaries_parent.u)[1], size(boundaries_parent.u)[2],
                          size(boundaries.node_indices)[end]))

    return boundaries
end

# Extract mortars that are entirely within the mesh view (all neighbor elements in view).
# Mortars that cross the view boundary are handled separately as "coupled mortars".
function extract_mortars(mesh::P4estMeshView, mortars_parent)
    # In 2D: mortars have 3 neighbors [small_1, small_2, large]
    # In 3D: mortars have 5 neighbors [small_1, small_2, small_3, small_4, large]
    n_neighbors = size(mortars_parent.neighbor_ids, 1)

    # Identify mortars where ALL neighbors are in this mesh view
    mask = BitArray(undef, nmortars(mortars_parent))
    for mortar in 1:nmortars(mortars_parent)
        all_in_view = true
        for pos in 1:n_neighbors
            if !(mortars_parent.neighbor_ids[pos, mortar] in mesh.cell_ids)
                all_in_view = false
                break
            end
        end
        mask[mortar] = all_in_view
    end

    n_mortars_view = sum(mask)

    # Create deepcopy and resize
    mortars = deepcopy(mortars_parent)
    resize!(mortars, n_mortars_view)

    if n_mortars_view == 0
        return mortars
    end

    # Copy relevant entries and convert to local indices
    mortar_indices = findall(mask)
    for (new_idx, old_idx) in enumerate(mortar_indices)
        for pos in 1:n_neighbors
            global_id = mortars_parent.neighbor_ids[pos, old_idx]
            mortars.neighbor_ids[pos, new_idx] = global_element_id_to_local(global_id, mesh)
        end
        # node_indices has shape (2, n_mortars): row 1 = small face, row 2 = large face.
        # Use column indexing to copy both entries correctly.
        mortars.node_indices[:, new_idx] = mortars_parent.node_indices[:, old_idx]
    end

    # Note: mortars.u arrays are already resized by resize!(mortars, n_mortars_view)
    # and will be filled with correct values during prolong2mortars!

    return mortars
end

# Extract the ids of the neighboring elements using the global indexing of the parent mesh.
function extract_neighbor_ids_global(mesh::P4estMeshView, boundaries_parent,
                                     interfaces_parent,
                                     boundaries, elements_parent)
    # Determine the global indices of the boundaring elements.
    neighbor_ids_global = zero.(boundaries.neighbor_ids)
    for (idx, id) in enumerate(boundaries.neighbor_ids)
        global_id = mesh.cell_ids[id]
        # Find this id in the parent's interfaces.
        for interface in eachindex(interfaces_parent.neighbor_ids[1, :])
            if global_id == interfaces_parent.neighbor_ids[1, interface] ||
               global_id == interfaces_parent.neighbor_ids[2, interface]
                if global_id == interfaces_parent.neighbor_ids[1, interface]
                    matching_boundary = 1
                else
                    matching_boundary = 2
                end
                # Check if interfaces with this id have the right name/node_indices.
                if boundaries.name[idx] ==
                   node_indices_to_name(interfaces_parent.node_indices[matching_boundary,
                                                                       interface])
                    if global_id == interfaces_parent.neighbor_ids[1, interface]
                        neighbor_ids_global[idx] = interfaces_parent.neighbor_ids[2,
                                                                                  interface]
                    else
                        neighbor_ids_global[idx] = interfaces_parent.neighbor_ids[1,
                                                                                  interface]
                    end
                end
            end
        end

        # Find this id in the parent's boundaries and match to opposite side.
        # Use coordinate-based matching to handle non-uniform refinement.
        for (parent_idx, boundary) in enumerate(boundaries_parent.neighbor_ids)
            if global_id == boundary
                # Check if boundaries with this id have the right name/node_indices.
                if boundaries.name[idx] == boundaries_parent.name[parent_idx]
                    # Find matching element on opposite boundary based on coordinate overlap.
                    opposite_name = get_opposite_boundary_name(boundaries_parent.name[parent_idx])
                    if opposite_name !== nothing
                        neighbor_ids_global[idx] = find_matching_boundary_element(
                            global_id, boundaries_parent.name[parent_idx],
                            opposite_name, boundaries_parent, elements_parent)
                    end
                end
            end
        end
    end

    return neighbor_ids_global
end

# Get the opposite boundary name for periodic-like coupling
function get_opposite_boundary_name(name::Symbol)
    if name == :x_neg
        return :x_pos
    elseif name == :x_pos
        return :x_neg
    elseif name == :y_neg
        return :y_pos
    elseif name == :y_pos
        return :y_neg
    else
        return nothing
    end
end

# Find the element on the opposite boundary that best matches the given element's position.
# For x boundaries, match by y-coordinate; for y boundaries, match by x-coordinate.
function find_matching_boundary_element(element_id, boundary_name, opposite_name,
                                        boundaries_parent, elements_parent)
    # Get elements on the opposite boundary
    opposite_element_ids = boundaries_parent.neighbor_ids[boundaries_parent.name .== opposite_name]

    if isempty(opposite_element_ids)
        return 0  # No matching element found
    end

    # Get the center coordinate of our element (perpendicular to boundary direction)
    # For x boundaries, we match by y; for y boundaries, we match by x
    if boundary_name in (:x_neg, :x_pos)
        # Match by y-coordinate
        coord_idx = 2  # y-coordinate
    else
        # Match by x-coordinate
        coord_idx = 1  # x-coordinate
    end

    # Get center coordinate of source element
    # node_coordinates has shape (ndims, nnodes, nnodes, nelements)
    nnodes = size(elements_parent.node_coordinates, 2)
    center_node = (nnodes + 1) ÷ 2
    source_coord = elements_parent.node_coordinates[coord_idx, center_node, center_node,
                                                    element_id]

    # Find the element on the opposite side with the closest matching coordinate
    best_match = opposite_element_ids[1]
    best_distance = Inf

    for opp_id in opposite_element_ids
        opp_coord = elements_parent.node_coordinates[coord_idx, center_node, center_node,
                                                     opp_id]
        distance = abs(source_coord - opp_coord)
        if distance < best_distance
            best_distance = distance
            best_match = opp_id
        end
    end

    return best_match
end

"""
    extract_coupled_mortars(mesh::P4estMeshView, mortars_parent)

Extract mortars from parent mesh that lie on the boundary of this mesh view.
A mortar is at the view boundary if its large element and small elements
are split across the view boundary.

Returns indices of mortars that cross the view boundary and information about
which elements are local vs. remote.
"""
function extract_coupled_mortars(mesh::P4estMeshView, mortars_parent)
    # For minimal prototype: find mortars where elements are split across view boundary
    coupled_mortar_indices = Int[]
    local_neighbor_ids_list = Vector{Int}[]
    local_neighbor_positions_list = Vector{Int}[]
    global_neighbor_ids_list = Vector{Int}[]

    # In 2D: mortars have 3 neighbors [small_1, small_2, large]
    n_small = 2^(ndims(mesh) - 1)  # 2 in 2D, 4 in 3D

    for mortar_id in 1:nmortars(mortars_parent)
        # Get neighbor IDs from parent mortar
        neighbor_ids = mortars_parent.neighbor_ids[:, mortar_id]

        large_id = neighbor_ids[end]
        small_ids = neighbor_ids[1:n_small]

        # Check if mortar crosses view boundary
        large_in_view = large_id in mesh.cell_ids
        small_in_view = [id in mesh.cell_ids for id in small_ids]

        # Mortar is "coupled" if some but not all elements are in this view.
        # This covers all cross-boundary cases:
        #   - Large in view, no small in view
        #   - Large not in view, some/all small in view
        #   - Large in view, some but not all small in view
        some_in_view = large_in_view || any(small_in_view)
        all_in_view = large_in_view && all(small_in_view)
        if some_in_view && !all_in_view
            push!(coupled_mortar_indices, mortar_id)

            # Collect locally available elements
            local_neighbor_ids = Int[]
            local_neighbor_positions = Int[]

            # Add small elements that are in this view
            for (pos, (small_id, in_view)) in enumerate(zip(small_ids, small_in_view))
                if in_view
                    push!(local_neighbor_ids, global_element_id_to_local(small_id, mesh))
                    push!(local_neighbor_positions, pos)
                end
            end

            # Add large element if in this view
            if large_in_view
                push!(local_neighbor_ids, global_element_id_to_local(large_id, mesh))
                push!(local_neighbor_positions, n_small + 1)  # 3 in 2D, 5 in 3D
            end

            # Store ALL global neighbor IDs for this mortar (needed for cross-view data exchange)
            # Order: [small_1, small_2, large] in 2D
            global_neighbor_ids = vcat(collect(small_ids), [large_id])

            push!(local_neighbor_ids_list, local_neighbor_ids)
            push!(local_neighbor_positions_list, local_neighbor_positions)
            push!(global_neighbor_ids_list, global_neighbor_ids)
        end
    end

    return coupled_mortar_indices, local_neighbor_ids_list,
           local_neighbor_positions_list, global_neighbor_ids_list
end

"""
    populate_coupled_mortars!(coupled_mortars, mesh, mortars_parent, elements_parent,
                             coupled_mortar_indices, local_neighbor_ids_list,
                             local_neighbor_positions_list, global_neighbor_ids_list,
                             dg)

Populate the coupled mortar container with data from parent mortars.
This includes computing normal directions from the parent elements' contravariant vectors.
"""
function populate_coupled_mortars!(coupled_mortars, mesh, mortars_parent, elements_parent,
                                  coupled_mortar_indices,
                                  local_neighbor_ids_list,
                                  local_neighbor_positions_list,
                                  global_neighbor_ids_list,
                                  dg)
    n_coupled = length(coupled_mortar_indices)

    if n_coupled == 0
        return coupled_mortars
    end

    # Resize container
    resize!(coupled_mortars, n_coupled)

    n_nodes = nnodes(dg)
    index_range = eachnode(dg)

    # Fill container with data
    for (idx, mortar_id) in enumerate(coupled_mortar_indices)
        coupled_mortars.local_neighbor_ids[idx] = local_neighbor_ids_list[idx]
        coupled_mortars.local_neighbor_positions[idx] = local_neighbor_positions_list[idx]
        coupled_mortars.global_neighbor_ids[idx] = global_neighbor_ids_list[idx]

        # Copy node indices from parent mortar
        small_indices = mortars_parent.node_indices[1, mortar_id]
        large_indices = mortars_parent.node_indices[2, mortar_id]
        coupled_mortars.node_indices[1, idx] = small_indices
        coupled_mortars.node_indices[2, idx] = large_indices

        # Compute normal directions from the contravariant vectors
        # These are needed for the flux computation
        small_direction = indices2direction(small_indices)
        large_direction = indices2direction(large_indices)

        i_small_start, i_small_step = index_to_start_step_2d(small_indices[1], index_range)
        j_small_start, j_small_step = index_to_start_step_2d(small_indices[2], index_range)

        i_large_start, i_large_step = index_to_start_step_2d(large_indices[1], index_range)
        j_large_start, j_large_step = index_to_start_step_2d(large_indices[2], index_range)

        # Get element IDs from parent mortar
        small_element_ids = mortars_parent.neighbor_ids[1:2, mortar_id]
        large_element_id = mortars_parent.neighbor_ids[3, mortar_id]

        # Compute normals for small elements (positions 1 and 2)
        for position in 1:2
            element = small_element_ids[position]
            i_small = i_small_start
            j_small = j_small_start

            for node in 1:n_nodes
                # Compute normal direction using parent element's contravariant vectors
                normal = get_normal_direction(small_direction,
                                              elements_parent.contravariant_vectors,
                                              i_small, j_small, element)

                # Store in coupled mortar container
                for dim in 1:ndims(mesh)
                    coupled_mortars.normal_directions[dim, node, position, idx] = normal[dim]
                end

                i_small += i_small_step
                j_small += j_small_step
            end
        end

        # Note: Position 3 (large element) normals are not computed here
        # The mortar flux is computed from the small elements' perspective only
    end

    return coupled_mortars
end

# Translate the interface indices into boundary names.
function node_indices_to_name(node_index)
    if node_index == (:end, :i_forward)
        return :x_pos
    elseif node_index == (:begin, :i_forward)
        return :x_neg
    elseif node_index == (:i_forward, :end)
        return :y_pos
    elseif node_index == (:i_forward, :begin)
        return :y_neg
    else
        error("Unknown node index: $node_index")
    end
end

# Convert a global cell id to a local cell id in the mesh view.
function global_element_id_to_local(id::Int, mesh::P4estMeshView)
    # Find the index of the cell id in the mesh view
    local_id = findfirst(==(id), mesh.cell_ids)

    return local_id
end

# Convert an array of global cell ids to a local cell id in the mesh view.
function global_element_id_to_local(id::AbstractArray, mesh::P4estMeshView)
    # Find the index of the cell id in the mesh view
    local_id = zeros(Int, length(id))
    for i in eachindex(id)
        local_id[i] = global_element_id_to_local(id[i], mesh)
    end

    return local_id
end

# Does not save the mesh itself to an HDF5 file. Instead saves important attributes
# of the mesh, like its size and the type of boundary mapping function.
# Then, within Trixi2Vtk, the P4estMeshView and its node coordinates are reconstructured from
# these attributes for plotting purposes
# | Warning: This overwrites any existing mesh file, either for a mesh view or parent mesh.
function save_mesh_file(mesh::P4estMeshView, output_directory; system = "",
                        timestep = 0)
    # Create output directory (if it does not exist)
    mkpath(output_directory)

    # Determine file name based on existence of meaningful time step
    filename = joinpath(output_directory,
                        @sprintf("mesh_%s_%09d.h5", system, timestep))
    p4est_filename = @sprintf("p4est_%s_data_%09d", system, timestep)

    p4est_file = joinpath(output_directory, p4est_filename)

    # Save the complete connectivity and `p4est` data to disk.
    save_p4est!(p4est_file, mesh.parent.p4est)

    # Open file (clobber existing content)
    h5open(filename, "w") do file
        # Add context information as attributes
        attributes(file)["mesh_type"] = get_name(mesh)
        attributes(file)["ndims"] = ndims(mesh)
        attributes(file)["p4est_file"] = p4est_filename
        attributes(file)["cell_ids"] = mesh.cell_ids

        file["tree_node_coordinates"] = mesh.parent.tree_node_coordinates
        file["nodes"] = Vector(mesh.parent.nodes) # the mesh uses `SVector`s for the nodes
        # to increase the runtime performance
        # but HDF5 can only handle plain arrays
        file["boundary_names"] = mesh.parent.boundary_names .|> String
        return nothing
    end

    return filename
end

# Interpolate tree_node_coordinates to each quadrant at the specified nodes
# Note: This is a copy of the corresponding function in src/solvers/dgsem_p4est/containers_2d.jl,
#       with modifications to skip cells not part of the mesh view
function calc_node_coordinates!(node_coordinates,
                                mesh::P4estMeshView{2, NDIMS_AMBIENT},
                                nodes::AbstractVector) where {NDIMS_AMBIENT}
    # We use `StrideArray`s here since these buffers are used in performance-critical
    # places and the additional information passed to the compiler makes them faster
    # than native `Array`s.
    tmp1 = StrideArray(undef, real(mesh),
                       StaticInt(NDIMS_AMBIENT), static_length(nodes),
                       static_length(mesh.parent.nodes))
    matrix1 = StrideArray(undef, real(mesh),
                          static_length(nodes), static_length(mesh.parent.nodes))
    matrix2 = similar(matrix1)
    baryweights_in = barycentric_weights(mesh.parent.nodes)

    # Macros from `p4est`
    p4est_root_len = 1 << P4EST_MAXLEVEL
    p4est_quadrant_len(l) = 1 << (P4EST_MAXLEVEL - l)

    trees = unsafe_wrap_sc(p4est_tree_t, mesh.parent.p4est.trees)

    mesh_view_cell_id = 0
    for tree_id in eachindex(trees)
        tree_offset = trees[tree_id].quadrants_offset
        quadrants = unsafe_wrap_sc(p4est_quadrant_t, trees[tree_id].quadrants)

        for i in eachindex(quadrants)
            parent_mesh_cell_id = tree_offset + i
            if !(parent_mesh_cell_id in mesh.cell_ids)
                # This cell is not part of the mesh view, thus skip it
                continue
            end
            mesh_view_cell_id += 1

            quad = quadrants[i]

            quad_length = p4est_quadrant_len(quad.level) / p4est_root_len

            nodes_out_x = 2 * (quad_length * 1 / 2 * (nodes .+ 1) .+
                           quad.x / p4est_root_len) .- 1
            nodes_out_y = 2 * (quad_length * 1 / 2 * (nodes .+ 1) .+
                           quad.y / p4est_root_len) .- 1
            polynomial_interpolation_matrix!(matrix1, mesh.parent.nodes, nodes_out_x,
                                             baryweights_in)
            polynomial_interpolation_matrix!(matrix2, mesh.parent.nodes, nodes_out_y,
                                             baryweights_in)

            multiply_dimensionwise!(view(node_coordinates, :, :, :, mesh_view_cell_id),
                                    matrix1, matrix2,
                                    view(mesh.parent.tree_node_coordinates, :, :, :,
                                         tree_id),
                                    tmp1)
        end
    end

    return node_coordinates
end

@inline mpi_parallel(mesh::P4estMeshView) = False()
end # @muladd
