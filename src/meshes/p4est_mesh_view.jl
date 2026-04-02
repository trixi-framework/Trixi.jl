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
@inline Base.real(::P4estMeshView{NDIMS, NDIMS_AMBIENT, RealT}) where {NDIMS,
NDIMS_AMBIENT,
RealT} = RealT

@inline ncells(mesh::P4estMeshView) = length(mesh.cell_ids)

# Extract interfaces, boundaries and parent element ids from the neighbors.
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

    # Get the parent element ids of the neighbors.
    neighbor_ids_parent = extract_neighbor_ids_parent(mesh, boundaries_parent,
                                                      interfaces_parent,
                                                      boundaries)

    return elements, interfaces, boundaries, mortars, neighbor_ids_parent
end

# Remove all interfaces that have a tuple of neighbor_ids where at least one is
# not part of this mesh view, i.e. mesh.cell_ids, and return the new interface container.
function extract_interfaces(mesh::P4estMeshView, interfaces_parent)
    # Identify interfaces that need to be retained
    mask = BitArray(undef, ninterfaces(interfaces_parent))
    # Loop over all interfaces (index 2).
    for interface in 1:size(interfaces_parent.neighbor_ids)[2]
        mask[interface] = (interfaces_parent.neighbor_ids[1,
                           interface] in mesh.cell_ids) &&
                          (interfaces_parent.neighbor_ids[2,
                           interface] in mesh.cell_ids)
    end

    # Create deepcopy to get completely independent interfaces container
    interfaces = deepcopy(interfaces_parent)
    resize!(interfaces, sum(mask))

    # Copy relevant entries from parent mesh
    @views interfaces.u .= interfaces_parent.u[.., mask]
    @views interfaces.node_indices .= interfaces_parent.node_indices[.., mask]
    @views neighbor_ids = interfaces_parent.neighbor_ids[.., mask]

    # Transform the parent indices into view indices.
    interfaces.neighbor_ids = zeros(Int, size(neighbor_ids))
    for interface in 1:size(neighbor_ids)[2]
        interfaces.neighbor_ids[1, interface] = findall(id -> id ==
                                                              neighbor_ids[1,
                                                                           interface],
                                                        mesh.cell_ids)[1]
        interfaces.neighbor_ids[2, interface] = findall(id -> id ==
                                                              neighbor_ids[2,
                                                                           interface],
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
    for boundary in 1:nboundaries(boundaries_parent)
        mask[boundary] = boundaries_parent.neighbor_ids[boundary] in mesh.cell_ids
    end
    boundaries.neighbor_ids = parent_cell_id_to_view(boundaries_parent.neighbor_ids[mask],
                                                     mesh)
    boundaries.name = boundaries_parent.name[mask]
    boundaries.node_indices = boundaries_parent.node_indices[mask]

    # Add new boundaries that were interfaces of the parent mesh.
    # Loop over all interfaces (index 2).
    for interface in 1:ninterfaces(interfaces_parent)
        # Create new boundary if exactly one of the neighbor cells is in the mesh view ("exclusive or" with ⊻)
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
                  parent_cell_id_to_view(neighbor_id, mesh))
            # Update the boundary names to reflect where the neighboring cell is
            # relative to this one, i.e. left, right, up, down.
            # In 3d one would need to add the third dimension.
            if (interfaces_parent.node_indices[view_idx, interface] ==
                (:end, :i_forward))
                push!(boundaries.name, :x_pos)
            elseif (interfaces_parent.node_indices[view_idx, interface] ==
                    (:begin, :i_forward))
                push!(boundaries.name, :x_neg)
            elseif (interfaces_parent.node_indices[view_idx, interface] ==
                    (:i_forward, :end))
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
    n_dims = ndims(boundaries)
    n_nodes = size(boundaries.u, 2)
    n_variables = size(boundaries.u, 1)
    capacity = length(boundaries.neighbor_ids)

    resize!(boundaries._u, n_variables * n_nodes^(n_dims - 1) * capacity)
    boundaries.u = unsafe_wrap(Array, pointer(boundaries._u),
                               (n_variables, ntuple(_ -> n_nodes, n_dims - 1)...,
                                capacity))

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
            mortars.neighbor_ids[pos, new_idx] = parent_cell_id_to_view(global_id, mesh)
        end
        # node_indices has shape (2, n_mortars): row 1 = small face, row 2 = large face.
        # Use column indexing to copy both entries correctly.
        mortars.node_indices[:, new_idx] = mortars_parent.node_indices[:, old_idx]
    end

    # Note: mortars.u arrays are already resized by resize!(mortars, n_mortars_view)
    # and will be filled with correct values during prolong2mortars!

    return mortars
end

# Extract the ids of the neighboring elements using the parent mesh indexing.
# For every boundary of the mesh view find the neighboring cell id in global (parent) indexing.
# Such neighboring cells are either inside the domain and have an interface
# in the parent mesh, or they are physical boundaries for which we then
# construct a periodic coupling by assigning as neighbor id the cell id
# on the other end of the domain.
function extract_neighbor_ids_parent(mesh::P4estMeshView,
                                     boundaries_parent, interfaces_parent,
                                     boundaries)
    # Determine the parent indices of the neighboring elements.
    neighbor_ids_parent = similar(boundaries.neighbor_ids)
    for (idx, id) in enumerate(boundaries.neighbor_ids)
        parent_id = mesh.cell_ids[id]
        # Find this id in the parent's interfaces.
        for interface in eachindex(interfaces_parent.neighbor_ids[1, :])
            if (parent_id == interfaces_parent.neighbor_ids[1, interface] ||
                parent_id == interfaces_parent.neighbor_ids[2, interface])
                if parent_id == interfaces_parent.neighbor_ids[1, interface]
                    matching_boundary = 1
                else
                    matching_boundary = 2
                end
                # Check if interfaces with this id have the right name/node_indices.
                if (boundaries.name[idx] ==
                    node_indices_to_name(interfaces_parent.node_indices[matching_boundary,
                                                                        interface]))
                    if parent_id == interfaces_parent.neighbor_ids[1, interface]
                        neighbor_ids_parent[idx] = interfaces_parent.neighbor_ids[2,
                                                                                  interface]
                    else
                        neighbor_ids_parent[idx] = interfaces_parent.neighbor_ids[1,
                                                                                  interface]
                    end
                end
            end
        end

        # Find this id in the parent's boundaries.
        parent_xneg_cell_ids = boundaries_parent.neighbor_ids[boundaries_parent.name .== :x_neg]
        parent_xpos_cell_ids = boundaries_parent.neighbor_ids[boundaries_parent.name .== :x_pos]
        parent_yneg_cell_ids = boundaries_parent.neighbor_ids[boundaries_parent.name .== :y_neg]
        parent_ypos_cell_ids = boundaries_parent.neighbor_ids[boundaries_parent.name .== :y_pos]
        for (parent_idx, boundary) in enumerate(boundaries_parent.neighbor_ids)
            if parent_id == boundary
                # Check if boundaries with this id have the right name/node_indices.
                if boundaries.name[idx] == boundaries_parent.name[parent_idx]
                    # Make the coupling periodic.
                    if boundaries_parent.name[parent_idx] == :x_neg
                        neighbor_ids_parent[idx] = parent_xpos_cell_ids[findfirst(parent_xneg_cell_ids .==
                                                                                  boundary)]
                    elseif boundaries_parent.name[parent_idx] == :x_pos
                        neighbor_ids_parent[idx] = parent_xneg_cell_ids[findfirst(parent_xpos_cell_ids .==
                                                                                  boundary)]
                    elseif boundaries_parent.name[parent_idx] == :y_neg
                        neighbor_ids_parent[idx] = parent_ypos_cell_ids[findfirst(parent_yneg_cell_ids .==
                                                                                  boundary)]
                    elseif boundaries_parent.name[parent_idx] == :y_pos
                        neighbor_ids_parent[idx] = parent_yneg_cell_ids[findfirst(parent_ypos_cell_ids .==
                                                                                  boundary)]
                    else
                        error("Unknown boundary name: $(boundaries_parent.name[parent_idx])")
                    end
                end
            end
        end
    end

    return neighbor_ids_parent
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
                    push!(local_neighbor_ids, parent_cell_id_to_view(small_id, mesh))
                    push!(local_neighbor_positions, pos)
                end
            end

            # Add large element if in this view
            if large_in_view
                push!(local_neighbor_ids, parent_cell_id_to_view(large_id, mesh))
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
# This works only in 2d currently.
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

# Convert a parent cell id to a view cell id in the mesh view.
function parent_cell_id_to_view(id::Integer, mesh::P4estMeshView)
    # Find the index of the cell id in the mesh view.
    # We use findfirst rather than searchsortedfirst to handle unsorted cell_ids correctly.
    view_id = findfirst(==(id), mesh.cell_ids)

    return view_id
end

# Convert an array of parent cell ids to view cell ids in the mesh view.
function parent_cell_id_to_view(ids::AbstractArray, mesh::P4estMeshView)
    view_id = zeros(Int, length(ids))
    for i in eachindex(ids)
        view_id[i] = parent_cell_id_to_view(ids[i], mesh)
    end
    return view_id
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

    filename = joinpath(output_directory, "mesh.h5")
    p4est_filename = "p4est_data"
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

    # Build a lookup from global parent cell ID → local view cell ID.
    # This respects the ordering of mesh.cell_ids (which may not be sorted),
    # ensuring consistency with extract_p4est_mesh_view which copies data in
    # mesh.cell_ids order.
    cell_id_to_local = Dict(id => k for (k, id) in enumerate(mesh.cell_ids))

    for tree_id in eachindex(trees)
        tree_offset = trees[tree_id].quadrants_offset
        quadrants = unsafe_wrap_sc(p4est_quadrant_t, trees[tree_id].quadrants)

        for i in eachindex(quadrants)
            parent_mesh_cell_id = tree_offset + i
            local_id = get(cell_id_to_local, parent_mesh_cell_id, 0)
            if local_id == 0
                # This cell is not part of the mesh view, thus skip it
                continue
            end

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

            multiply_dimensionwise!(view(node_coordinates, :, :, :, local_id),
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
