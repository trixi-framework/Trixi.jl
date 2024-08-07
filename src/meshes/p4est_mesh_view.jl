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
    indices_min::NTuple{NDIMS, Int}
    indices_max::NTuple{NDIMS, Int}
    trees_per_dimension::NTuple{NDIMS, Int}
end

# function P4estMeshView(parent::P4estMesh{NDIMS, RealT}, view_cells::Array{Bool}) where {NDIMS, RealT}
function P4estMeshView(parent::P4estMesh{NDIMS, RealT};
                       indices_min = ntuple(_ -> 1, Val(NDIMS)),
                       indices_max = size(parent),
                       coordinates_min = nothing, coordinates_max = nothing,
                       periodicity = (true, true)) where {NDIMS, RealT}
    trees_per_dimension = indices_max .- indices_min .+ (1, 1)

    @assert indices_min <= indices_max
    @assert all(indices_min .> 0)
    @assert prod(trees_per_dimension) <= size(parent)

    ghost = ghost_new_p4est(parent.p4est)
    ghost_pw = PointerWrapper(ghost)

    # Extract mapping
    if isnothing(coordinates_min)
        coordinates_min = minimum(parent.tree_node_coordinates)
    end
    if isnothing(coordinates_max)
        coordinates_max = maximum(parent.tree_node_coordinates)
    end

    mapping = coordinates2mapping(coordinates_min, coordinates_max)

    tree_node_coordinates = Array{RealT, NDIMS + 2}(undef, NDIMS,
                                                    ntuple(_ -> length(parent.nodes),
                                                           NDIMS)...,
                                                    prod(trees_per_dimension))
    calc_tree_node_coordinates!(tree_node_coordinates, parent.nodes, mapping,
                                trees_per_dimension)

    connectivity = connectivity_structured(trees_per_dimension..., periodicity)

    # TODO: The initial refinement level of 1 should not be hard-coded.
    p4est = new_p4est(connectivity, 1)
    p4est_pw = PointerWrapper(p4est)

    # Non-periodic boundaries
    boundary_names = fill(Symbol("---"), 2 * NDIMS, prod(trees_per_dimension))

    structured_boundary_names!(boundary_names, trees_per_dimension, periodicity)

    return P4estMeshView{NDIMS, eltype(parent.tree_node_coordinates),
                         typeof(parent.is_parallel),
                         typeof(p4est_pw), typeof(parent.ghost), NDIMS + 2,
                         length(parent.nodes)}(PointerWrapper(p4est),
                                               parent.is_parallel,
                                               parent.ghost,
                                               tree_node_coordinates,
                                               parent.nodes, boundary_names,
                                               parent.current_filename,
                                               parent.unsaved_changes,
                                               parent.p4est_partition_allow_for_coarsening,
                                               parent, indices_min, indices_max, trees_per_dimension)
end

@inline Base.ndims(::P4estMeshView{NDIMS}) where {NDIMS} = NDIMS
@inline Base.real(::P4estMeshView{NDIMS, RealT}) where {NDIMS, RealT} = RealT
@inline function ntrees(mesh::P4estMeshView)
    return mesh.p4est.trees.elem_count[]
end
@inline ncellsglobal(mesh::P4estMeshView) = Int(mesh.p4est.global_num_quadrants[])
Base.axes(mesh::P4estMeshView) = map(Base.OneTo, size(mesh))
Base.axes(mesh::P4estMeshView, i) = Base.OneTo(size(mesh, i))
# Base.size(mesh::P4estMeshView) = size(mesh.tree_node_coordinates)[end]
Base.size(mesh::P4estMeshView) = mesh.trees_per_dimension
function Base.size(mesh::P4estMeshView, i)
    @unpack indices_min, indices_max = mesh
    return indices_max[i] - indices_min[i] + 1
end

function Base.show(io::IO, mesh::P4estMeshView)
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

function calc_node_coordinates!(node_coordinates,
                                mesh::P4estMeshView{2},
                                nodes::AbstractVector)
    # We use `StrideArray`s here since these buffers are used in performance-critical
    # places and the additional information passed to the compiler makes them faster
    # than native `Array`s.
    tmp1 = StrideArray(undef, real(mesh),
                       StaticInt(2), static_length(nodes), static_length(mesh.nodes))
    matrix1 = StrideArray(undef, real(mesh),
                          static_length(nodes), static_length(mesh.nodes))
    matrix2 = similar(matrix1)
    baryweights_in = barycentric_weights(mesh.nodes)

    # Macros from `p4est`
    p4est_root_len = 1 << P4EST_MAXLEVEL
    p4est_quadrant_len(l) = 1 << (P4EST_MAXLEVEL - l)

    trees = unsafe_wrap_sc(p4est_tree_t, mesh.p4est.trees)

    for tree in eachindex(trees)
        offset = trees[tree].quadrants_offset
        quadrants = unsafe_wrap_sc(p4est_quadrant_t, trees[tree].quadrants)

        for i in eachindex(quadrants)
            element = offset + i
            quad = quadrants[i]

            quad_length = p4est_quadrant_len(quad.level) / p4est_root_len

            nodes_out_x = 2 * (quad_length * 1 / 2 * (nodes .+ 1) .+
                           quad.x / p4est_root_len) .- 1
            nodes_out_y = 2 * (quad_length * 1 / 2 * (nodes .+ 1) .+
                           quad.y / p4est_root_len) .- 1
            polynomial_interpolation_matrix!(matrix1, mesh.nodes, nodes_out_x,
                                             baryweights_in)
            polynomial_interpolation_matrix!(matrix2, mesh.nodes, nodes_out_y,
                                             baryweights_in)

            multiply_dimensionwise!(view(node_coordinates, :, :, :, element),
                                    matrix1, matrix2,
                                    view(mesh.tree_node_coordinates, :, :, :, tree),
                                    tmp1)
        end
    end

    return node_coordinates
end

# # inlined version of the boundary flux calculation along a physical interface
# @inline function calc_boundary_flux!(surface_flux_values, t, boundary_condition,
#                                      mesh::Union{P4estMeshView{2}},
#                                      nonconservative_terms::False, equations,
#                                      surface_integral, dg::DG, cache,
#                                      i_index, j_index,
#                                      node_index, direction_index, element_index,
#                                      boundary_index)
#     @unpack boundaries = cache
#     @unpack node_coordinates, contravariant_vectors = cache.elements
#     @unpack surface_flux = surface_integral
#
#     # Extract solution data from boundary container
#     u_inner = get_node_vars(boundaries.u, equations, dg, node_index, boundary_index)
#
#     # Outward-pointing normal direction (not normalized)
#     normal_direction = get_normal_direction(direction_index, contravariant_vectors,
#                                             i_index, j_index, element_index)
#
#     # Coordinates at boundary node
#     x = get_node_coords(node_coordinates, equations, dg, i_index, j_index,
#                         element_index)
#
#     println(i_index, " ", j_index, " ", node_index, " ", element_index, " ", boundary_index, " ", x)
#
#     flux_ = boundary_condition(u_inner, normal_direction, x, t, surface_flux, equations)
# #     flux_ = boundary_condition(u_inner, orientation, direction, cell_indices, surface_node_indices, surface_flux_function, equations)
#
#     # Copy flux to element storage in the correct orientation
#     for v in eachvariable(equations)
#         surface_flux_values[v, node_index, direction_index, element_index] = flux_[v]
#     end
# end

function save_mesh_file(mesh::P4estMeshView, output_directory; timestep = 0,
                        system = "")
    # Create output directory (if it does not exist)
    mkpath(output_directory)

    # Determine file name based on existence of meaningful time step
    if timestep > 0
        filename = joinpath(output_directory,
                            @sprintf("mesh_%s_%09d.h5", system, timestep))
        p4est_filename = @sprintf("p4est_data_%s_%09d", system, timestep)
    else
        filename = joinpath(output_directory, @sprintf("mesh_%s.h5", system))
        p4est_filename = @sprintf("p4est_data_%s", system)
    end

    p4est_file = joinpath(output_directory, p4est_filename)

    # Save the complete connectivity and `p4est` data to disk.
    save_p4est!(p4est_file, mesh.p4est)

    # Open file (clobber existing content)
    h5open(filename, "w") do file
        # Add context information as attributes
        attributes(file)["mesh_type"] = get_name(mesh.parent)
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
