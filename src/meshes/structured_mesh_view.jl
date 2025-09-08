# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

"""
    StructuredMeshView{NDIMS, RealT <: Real} <: AbstractMesh{NDIMS}

A view on a structured curved mesh.
"""
mutable struct StructuredMeshView{NDIMS, RealT <: Real} <: AbstractMesh{NDIMS}
    parent::StructuredMesh{NDIMS, RealT}
    cells_per_dimension::NTuple{NDIMS, Int}
    mapping::Any # Not relevant for performance
    mapping_as_string::String
    current_filename::String
    indices_min::NTuple{NDIMS, Int}
    indices_max::NTuple{NDIMS, Int}
    unsaved_changes::Bool
end

"""
    StructuredMeshView(parent; indices_min, indices_max)

Create a StructuredMeshView on a StructuredMesh parent.

# Arguments
- `parent`: the parent StructuredMesh.
- `indices_min`: starting indices of the parent mesh.
- `indices_max`: ending indices of the parent mesh.
"""
function StructuredMeshView(parent::StructuredMesh{NDIMS, RealT};
                            indices_min = ntuple(_ -> 1, Val(NDIMS)),
                            indices_max = size(parent)) where {NDIMS, RealT}
    @assert indices_min <= indices_max
    @assert all(indices_min .> 0)
    @assert indices_max <= size(parent)

    cells_per_dimension = indices_max .- indices_min .+ 1

    # Compute cell sizes `deltas`
    deltas = (parent.mapping.coordinates_max .- parent.mapping.coordinates_min) ./
             parent.cells_per_dimension
    # Calculate the domain boundaries.
    coordinates_min = parent.mapping.coordinates_min .+ deltas .* (indices_min .- 1)
    coordinates_max = parent.mapping.coordinates_min .+ deltas .* indices_max
    mapping = coordinates2mapping(coordinates_min, coordinates_max)
    mapping_as_string = """
        coordinates_min = $coordinates_min
        coordinates_max = $coordinates_max
        mapping = coordinates2mapping(coordinates_min, coordinates_max)
        """

    return StructuredMeshView{NDIMS, RealT}(parent, cells_per_dimension, mapping,
                                            mapping_as_string,
                                            parent.current_filename,
                                            indices_min, indices_max,
                                            parent.unsaved_changes)
end

# Check if mesh is periodic
function isperiodic(mesh::StructuredMeshView)
    @unpack parent = mesh
    return isperiodic(parent) && size(parent) == size(mesh)
end

function isperiodic(mesh::StructuredMeshView, dimension)
    @unpack parent, indices_min, indices_max = mesh
    return (isperiodic(parent, dimension) &&
            indices_min[dimension] == 1 &&
            indices_max[dimension] == size(parent, dimension))
end

@inline Base.ndims(::StructuredMeshView{NDIMS}) where {NDIMS} = NDIMS
@inline Base.real(::StructuredMeshView{NDIMS, RealT}) where {NDIMS, RealT} = RealT
function Base.size(mesh::StructuredMeshView)
    @unpack indices_min, indices_max = mesh
    return indices_max .- indices_min .+ 1
end
function Base.size(mesh::StructuredMeshView, i)
    @unpack indices_min, indices_max = mesh
    return indices_max[i] - indices_min[i] + 1
end
Base.axes(mesh::StructuredMeshView) = map(Base.OneTo, size(mesh))
Base.axes(mesh::StructuredMeshView, i) = Base.OneTo(size(mesh, i))

function calc_node_coordinates!(node_coordinates, element,
                                cell_x, cell_y, mapping,
                                mesh::StructuredMeshView{2},
                                basis)
    @unpack nodes = basis

    # Get cell length in reference mesh
    dx = 2 / size(mesh, 1)
    dy = 2 / size(mesh, 2)

    # Calculate node coordinates of reference mesh
    cell_x_offset = -1 + (cell_x - 1) * dx + dx / 2
    cell_y_offset = -1 + (cell_y - 1) * dy + dy / 2

    for j in eachnode(basis), i in eachnode(basis)
        # node_coordinates are the mapped reference node_coordinates
        node_coordinates[:, i, j, element] .= mapping(cell_x_offset + dx / 2 * nodes[i],
                                                      cell_y_offset + dy / 2 * nodes[j])
    end
end

# Does not save the mesh itself to an HDF5 file. Instead saves important attributes
# of the mesh, like its size and the type of boundary mapping function.
# Then, within Trixi2Vtk, the StructuredMesh and its node coordinates are reconstructured from
# these attributes for plotting purposes.
function save_mesh_file(mesh::StructuredMeshView, output_directory; system = "",
                        timestep = 0)
    # Create output directory (if it does not exist)
    mkpath(output_directory)

    filename = joinpath(output_directory, @sprintf("mesh_%s_%09d.h5", system, timestep))

    # Open file (clobber existing content)
    h5open(filename, "w") do file
        # Add context information as attributes
        attributes(file)["mesh_type"] = get_name(mesh)
        attributes(file)["ndims"] = ndims(mesh)
        attributes(file)["size"] = collect(size(mesh))
        attributes(file)["mapping"] = mesh.mapping_as_string
    end

    return filename
end
end # @muladd
