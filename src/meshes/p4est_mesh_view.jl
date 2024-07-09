@muladd begin
#! format: noindent

mutable struct P4estMeshView{NDIMS, RealT <: Real, IsParallel, P, Ghost, NDIMSP2,
                             NNODES} <:
               AbstractMesh{NDIMS}
    # Attributes from the original P4est mesh
    p4est       :: P # Either PointerWrapper{p4est_t} or PointerWrapper{p8est_t}
    is_parallel :: IsParallel
    ghost       :: Ghost # Either PointerWrapper{p4est_ghost_t} or PointerWrapper{p8est_ghost_t}
    # Coordinates at the nodes specified by the tensor product of `nodes` (NDIMS times).
    # This specifies the geometry interpolation for each tree.
    tree_node_coordinates::Array{RealT, NDIMSP2} # [dimension, i, j, k, tree]
    nodes::SVector{NNODES, RealT}
    boundary_names::Array{Symbol, 2}      # [face direction, tree]
    current_filename::String
    unsaved_changes::Bool
    p4est_partition_allow_for_coarsening::Bool

    # Attributes pertaining the views.
    parent::P4estMesh{NDIMS, RealT}
end

function P4estMeshView(parent::P4estMesh{NDIMS, RealT}) where {NDIMS, RealT}
    ghost = ghost_new_p4est(parent.p4est)
    ghost_pw = PointerWrapper(ghost)

    return P4estMeshView{NDIMS, eltype(parent.tree_node_coordinates),
                         typeof(parent.is_parallel),
                         typeof(parent.p4est), typeof(parent.ghost), NDIMS + 2,
                         length(parent.nodes)}(parent.p4est, parent.is_parallel,
                                               parent.ghost,
                                               parent.tree_node_coordinates,
                                               parent.nodes, parent.boundary_names,
                                               parent.current_filename,
                                               parent.unsaved_changes,
                                               parent.p4est_partition_allow_for_coarsening,
                                               parent)
end

# TODO: Check if this is still needed.
# At the end we will have every cell boundary with a boundary condition.
# Check if mesh is periodic
function isperiodic(mesh::P4estMeshView)
    @unpack parent = mesh
    return isperiodic(parent) && size(parent) == size(mesh)
end

function isperiodic(mesh::P4estMeshView, dimension)
    @unpack parent = mesh
    return (isperiodic(parent, dimension))
end

@inline Base.ndims(::P4estMeshView{NDIMS}) where {NDIMS} = NDIMS
@inline Base.real(::P4estMeshView{NDIMS, RealT}) where {NDIMS, RealT} = RealT
@inline function ntrees(mesh::P4estMeshView)
    return mesh.p4est.trees.elem_count[]
end
@inline ncellsglobal(mesh::P4estMeshView) = Int(mesh.p4est.global_num_quadrants[])
Base.axes(mesh::P4estMeshView) = map(Base.OneTo, size(mesh))
Base.axes(mesh::P4estMeshView, i) = Base.OneTo(size(mesh, i))

function Base.show(io::IO, mesh::P4estMesh)
    print(io, "P4estMeshView{", ndims(mesh), ", ", real(mesh), "}")
end

function Base.show(io::IO, ::MIME"text/plain", mesh::P4estMeshView)
    if get(io, :compact, false)
        show(io, mesh)
    else
        setup = [
            "#trees" => ntrees(mesh),
            "current #cells" => ncellsglobal(mesh),
            "polydeg" => length(mesh.nodes) - 1,
        ]
        summary_box(io,
                    "P4estMeshView{" * string(ndims(mesh)) * ", " * string(real(mesh)) *
                    "}", setup)
    end
end

function balance!(mesh::P4estMeshView{2}, init_fn = C_NULL)
    p4est_balance(mesh.p4est, P4EST_CONNECT_FACE, init_fn)
    # Due to a bug in `p4est`, the forest needs to be rebalanced twice sometimes
    # See https://github.com/cburstedde/p4est/issues/112
    p4est_balance(mesh.p4est, P4EST_CONNECT_FACE, init_fn)
end

@inline ncells(mesh::P4estMeshView) = Int(mesh.p4est.local_num_quadrants[])

function save_mesh_file(mesh::P4estMeshView, output_directory, timestep = 0;
                        system = "")
    # Create output directory (if it does not exist)
    mkpath(output_directory)

    # Determine file name based on existence of meaningful time step
    if timestep > 0
        filename = joinpath(output_directory,
                            @sprintf("mesh_%s_%06d.h5", system, timestep))
        p4est_filename = @sprintf("p4est_data_%s_%06d", system, timestep)
    else
        filename = joinpath(output_directory, "mesh.h5")
        p4est_filename = "p4est_data"
    end

    p4est_file = joinpath(output_directory, p4est_filename)

    # Save the complete connectivity and `p4est` data to disk.
    save_p4est!(p4est_file, mesh.p4est)

    # Open file (clobber existing content)
    h5open(filename, "w") do file
        # Add context information as attributes
        attributes(file)["mesh_type"] = get_name(mesh)
        attributes(file)["ndims"] = ndims(mesh)
        attributes(file)["p4est_file"] = p4est_filename

        file["tree_node_coordinates"] = mesh.tree_node_coordinates
        file["nodes"] = Vector(mesh.nodes) # the mesh uses `SVector`s for the nodes
        # to increase the runtime performance
        # but HDF5 can only handle plain arrays
        file["boundary_names"] = mesh.boundary_names .|> String
    end

    return filename
end
end # @muladd
