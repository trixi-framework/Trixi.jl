# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

"""
    P4estMeshView{NDIMS, NDIMS_AMBIENT, RealT <: Real, Parent} <: AbstractMesh{NDIMS}

A view on a p4est mesh.
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

Create a P4estMeshView on a P4estMesh parent.

# Arguments
- `parent`: the parent P4estMesh.
- `cell_ids`: array of cell ids that are part of this view.
"""
function P4estMeshView(parent::P4estMesh{NDIMS, NDIMS_AMBIENT, RealT},
                       cell_ids::Vector) where {NDIMS, NDIMS_AMBIENT, RealT}
    return P4estMeshView{NDIMS, NDIMS_AMBIENT, RealT, typeof(parent)}(parent, cell_ids,
                                                                      parent.unsaved_changes,
                                                                      parent.current_filename)
end

@inline Base.ndims(::P4estMeshView{NDIMS}) where {NDIMS} = NDIMS
@inline Base.real(::P4estMeshView{NDIMS, RealT}) where {NDIMS, RealT} = RealT
@inline ndims_ambient(::P4estMeshView{NDIMS}) where {NDIMS} = NDIMS

@inline balance!(::P4estMeshView) = nothing
@inline ncells(mesh::P4estMeshView) = length(mesh.cell_ids)

function extract_p4est_mesh_view(elements_parent,
                                 interfaces_parent,
                                 boundaries_parent,
                                 mortars_parent,
                                 mesh)
    elements = elements_parent
    elements.inverse_jacobian = elements_parent.inverse_jacobian[.., mesh.cell_ids]
    elements.jacobian_matrix = elements_parent.jacobian_matrix[.., mesh.cell_ids]
    elements.node_coordinates = elements_parent.node_coordinates[.., mesh.cell_ids]
    elements.contravariant_vectors = elements_parent.contravariant_vectors[..,
                                                                           mesh.cell_ids]
    elements.surface_flux_values = elements_parent.surface_flux_values[..,
                                                                       mesh.cell_ids]
    elements._inverse_jacobian = vec(elements.inverse_jacobian)
    elements._jacobian_matrix = vec(elements.jacobian_matrix)
    elements._node_coordinates = vec(elements.node_coordinates)
    elements._node_coordinates = vec(elements.node_coordinates)
    elements._surface_flux_values = vec(elements.surface_flux_values)
    interfaces = extract_interfaces(mesh, interfaces_parent)

    return elements, interfaces, boundaries_parent, mortars_parent
end

function extract_interfaces(mesh::P4estMeshView, interfaces_parent)
    """
    Remove all interfaces that have a tuple of neighbor_ids where at least one is
    not part of this meshview, i.e. mesh.cell_ids.
    """

    mask = BitArray(undef, size(interfaces_parent.neighbor_ids)[2])
    for interface in 1:size(interfaces_parent.neighbor_ids)[2]
        mask[interface] = (interfaces_parent.neighbor_ids[1, interface] in mesh.cell_ids) &&
                          (interfaces_parent.neighbor_ids[2, interface] in mesh.cell_ids)
    end
    interfaces = interfaces_parent
    interfaces.u = interfaces_parent.u[.., mask]
    interfaces.node_indices = interfaces_parent.node_indices[.., mask]
    neighbor_ids = interfaces_parent.neighbor_ids[.., mask]
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

    # Flatten the arrays.
    interfaces._u = vec(interfaces.u)
    interfaces._node_indices = vec(interfaces.node_indices)
    interfaces._neighbor_ids = vec(interfaces.neighbor_ids)

    return interfaces
end

# Does not save the mesh itself to an HDF5 file. Instead saves important attributes
# of the mesh, like its size and the type of boundary mapping function.
# Then, within Trixi2Vtk, the P4estMeshView and its node coordinates are reconstructured from
# these attributes for plotting purposes
function save_mesh_file(mesh::P4estMeshView, output_directory, timestep,
                        mpi_parallel::False)
    # Create output directory (if it does not exist)
    mkpath(output_directory)

    # Determine file name based on existence of meaningful time step
    if timestep > 0
        filename = joinpath(output_directory, @sprintf("mesh_%09d.h5", timestep))
        p4est_filename = @sprintf("p4est_data_%09d", timestep)
    else
        filename = joinpath(output_directory, "mesh.h5")
        p4est_filename = "p4est_data"
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

        file["tree_node_coordinates"] = mesh.parent.tree_node_coordinates[..,
                                                                          mesh.cell_ids]
        file["nodes"] = Vector(mesh.parent.nodes) # the mesh uses `SVector`s for the nodes
        # to increase the runtime performance
        # but HDF5 can only handle plain arrays
        file["boundary_names"] = mesh.parent.boundary_names .|> String
    end

    return filename
end

@inline mpi_parallel(mesh::P4estMeshView) = False()
end # @muladd