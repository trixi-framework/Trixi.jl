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
    parent::Parent
    cell_ids::Vector{Int}
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
    boundaries = extract_boundaries(mesh, boundaries_parent, interfaces_parent, interfaces)

    return elements, interfaces, boundaries, mortars_parent
end

# Remove all interfaces that have a tuple of neighbor_ids where at least one is
# not part of this meshview, i.e. mesh.cell_ids, and return the new interface container
function extract_interfaces(mesh::P4estMeshView, interfaces_parent)
    # Identify interfaces that need to be retained
    mask = BitArray(undef, ninterfaces(interfaces_parent))
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
function extract_boundaries(mesh::P4estMeshView, boundaries_parent, interfaces_parent, interfaces)
    # Remove all boundaries that are not part of this p4est mesh view.
    boundaries = deepcopy(boundaries_parent)
    mask = BitArray(undef, nboundaries(boundaries_parent))
    for boundary in 1:size(boundaries_parent.neighbor_ids)[1]
        mask[boundary] = boundaries_parent.neighbor_ids[boundary] in mesh.cell_ids
    end
    boundaries.neighbor_ids = global_element_id_to_local(boundaries_parent.neighbor_ids[mask], mesh)
    boundaries.name = boundaries_parent.name[mask]
    boundaries.node_indices = boundaries_parent.node_indices[mask]

    # Add new boundaries that were interfaces of the parent mesh.
    for interface in 1:size(interfaces_parent.neighbor_ids)[2]
        if ((interfaces_parent.neighbor_ids[1, interface] in mesh.cell_ids) &&
            !(interfaces_parent.neighbor_ids[2, interface] in mesh.cell_ids)) ||
            ((interfaces_parent.neighbor_ids[2, interface] in mesh.cell_ids) &&
            !(interfaces_parent.neighbor_ids[1, interface] in mesh.cell_ids))
            if interfaces_parent.neighbor_ids[1, interface] in mesh.cell_ids
                neighbor_id = interfaces_parent.neighbor_ids[1, interface]
                view_idx = 1
            else
                neighbor_id = interfaces_parent.neighbor_ids[2, interface]
                view_idx = 2
            end

            push!(boundaries.neighbor_ids, global_element_id_to_local(neighbor_id, mesh))
            if interfaces_parent.node_indices[view_idx, interface] == (:end, :i_forward)
                push!(boundaries.name, :x_pos)
            elseif interfaces_parent.node_indices[view_idx, interface] == (:begin, :i_forward)
                push!(boundaries.name, :x_neg)
            elseif interfaces_parent.node_indices[view_idx, interface] == (:i_forward, :end)
                push!(boundaries.name, :y_pos)
            else
                push!(boundaries.name, :y_neg)
            end

            push!(boundaries.node_indices, interfaces_parent.node_indices[view_idx, interface])
        end
    end

    boundaries.u = zeros(typeof(boundaries_parent.u).parameters[1],
                         (size(boundaries_parent.u)[1], size(boundaries_parent.u)[2], size(boundaries.node_indices)[end]))
    
    return boundaries
end

# Convert a global cell id to a local cell id in the mesh view.
function global_element_id_to_local(id::Int, mesh::P4estMeshView)
    # Find the index of the cell id in the mesh view
    local_id = findfirst(==(id), mesh.cell_ids)

    return local_id
end

# Convert a global cell id to a local cell id in the mesh view.
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
function save_mesh_file(mesh::P4estMeshView, output_directory; system = "", timestep = 0)
    # Create output directory (if it does not exist)
    mkpath(output_directory)

    # Determine file name based on existence of meaningful time step
    if timestep > 0
        filename = joinpath(output_directory, @sprintf("mesh_%s_%09d.h5", system, timestep))
        p4est_filename = @sprintf("p4est_%s_data_%09d", system, timestep)
    else
        filename = joinpath(output_directory, @sprintf("mesh_%s.h5", system))
        p4est_filename = @sprintf("p4est_%s_data", system)
    end

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
