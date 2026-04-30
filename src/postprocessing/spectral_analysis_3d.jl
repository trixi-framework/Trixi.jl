# By default, Julia/LLVM does not use fused multiply-add operations (FMAs)
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details
@muladd begin
#! format: noindent

"""
    compute_energy_spectrum(v1, v2, v3; normalize = true)

Compute an isotropic 1D kinetic energy spectrum from three 3D Cartesian velocity
components `v1`, `v2`, `v3`.
"""
function compute_energy_spectrum(v1::AbstractArray{<:Any, 3},
                                 v2::AbstractArray{<:Any, 3},
                                 v3::AbstractArray{<:Any, 3};
                                 normalize = true)

    # Compute the energy modes using FFTW
    energy_modes = 0.5 .* (abs2.(fft(v1)) .+ abs2.(fft(v2)) .+ abs2.(fft(v3)))
    if normalize
        energy_modes ./= length(energy_modes)^2
    end

    return radial_energy_spectrum(energy_modes)
end

# Multiple dispatch for handling tuples of velocity components
function compute_energy_spectrum(velocity_cartesian::NTuple{3, AbstractArray};
                                 normalize = true)
    compute_energy_spectrum(velocity_cartesian...;
                            normalize = normalize)
end

"""
    compute_energy_spectrum(u, mesh::TreeMesh{3}, equations, solver::DGSEM, cache;
                            normalize = true)

Compute the energy spectrum for a non-AMR 3D `TreeMesh`/`DGSEM` solution by first
interpolating from LGL nodes to a uniform Cartesian grid.
"""
function compute_energy_spectrum(u, mesh::TreeMesh{3},
                                 equations::AbstractCompressibleEulerEquations,
                                 solver::DGSEM, cache;
                                 normalize = true)
    primitive_variables = interpolate_lgl_to_uniform_cartesian(u, mesh, equations,
                                                               solver,
                                                               cache)
    rho = primitive_variables[1]
    density_weighted_velocity_1 = sqrt.(rho) .* primitive_variables[2]
    density_weighted_velocity_2 = sqrt.(rho) .* primitive_variables[3]
    density_weighted_velocity_3 = sqrt.(rho) .* primitive_variables[4]

    return compute_energy_spectrum(density_weighted_velocity_1,
                                   density_weighted_velocity_2,
                                   density_weighted_velocity_3; normalize)
end

function interpolate_lgl_to_uniform_cartesian(u, mesh::TreeMesh{3},
                                              equations::AbstractCompressibleEulerEquations,
                                              solver::DGSEM, cache)
    # Restrict to straightforward non AMR setups
    leaf_cell_ids = leaf_cells(mesh.tree)
    levels = mesh.tree.levels[leaf_cell_ids]
    if !all(==(first(levels)), levels)
        throw(ArgumentError("Non-uniform meshes are not supported yet"))
    end

    level = first(levels)
    cells_per_dimension = 2^level
    n_uniform_nodes = polydeg(solver) + 1
    grid_points_per_dimension = n_uniform_nodes * cells_per_dimension

    # `primitive_variables` stores primitive variables as `[rho, v1, v2, v3]` in 3D after
    # interpolation to the global Cartesian grid
    # Each entry has `grid_points_per_dimension` points in every coordinate direction
    primitive_grid_size = (grid_points_per_dimension, grid_points_per_dimension,
                           grid_points_per_dimension)
    primitive_variables = Vector{Array{real(solver), 3}}(undef, 4)
    for variable in eachindex(primitive_variables)
        primitive_variables[variable] = Array{real(solver)}(undef, primitive_grid_size)
    end

    # Interpolate from LGL nodes to cell-centered equidistant nodes in each element
    dx_reference = 2 / n_uniform_nodes
    nodes_out = collect(range(-1 + dx_reference / 2, 1 - dx_reference / 2,
                              length = n_uniform_nodes))
    vandermonde = polynomial_interpolation_matrix(get_nodes(solver.basis), nodes_out)

    # Element center coordinates determine where each interpolated block sits in the global grid
    center = reshape(mesh.tree.center_level_0, :, 1)
    normalized_coordinates = (mesh.tree.coordinates[:, leaf_cell_ids] .- center) ./
                             (mesh.tree.length_level_0 / 2)
    dx_global = 2 / (n_uniform_nodes * cells_per_dimension)

    for element in eachelement(solver, cache)
        # Gather local nodal values for all primitive variables in this element
        first_node = (1, 1, 1)
        u_node = cons2prim(get_node_vars(u, equations, solver, first_node..., element),
                           equations)

        element_primitive_values_size = (length(primitive_variables), nnodes(solver),
                                         nnodes(solver), nnodes(solver))
        element_primitive_values = Array{eltype(u_node)}(undef,
                                                         element_primitive_values_size)
        for node in CartesianIndices(Base.tail(size(element_primitive_values)))
            u_node = cons2prim(get_node_vars(u, equations, solver, Tuple(node)...,
                                             element),
                               equations)
            for variable in eachindex(primitive_variables)
                element_primitive_values[variable, Tuple(node)...] = u_node[variable]
            end
        end

        # Interpolate in each dimension using the tensor product structure
        interpolated = multiply_dimensionwise(vandermonde, element_primitive_values)

        first_index = Vector{Int}(undef, 3)
        for dim in 1:3
            lower_left = normalized_coordinates[dim, element] -
                         (n_uniform_nodes - 1) / 2 * dx_global
            first_index[dim] = round(Int,
                                     (lower_left - (-1 + dx_global / 2)) /
                                     dx_global) + 1
        end

        element_indices = (first_index[1]:(first_index[1] + n_uniform_nodes - 1),
                           first_index[2]:(first_index[2] + n_uniform_nodes - 1),
                           first_index[3]:(first_index[3] + n_uniform_nodes - 1))
        for variable in eachindex(primitive_variables)
            primitive_variables[variable][element_indices...] .= selectdim(interpolated,
                                                                           1,
                                                                           variable)
        end
    end

    return primitive_variables
end

"""
    compute_energy_spectrum(u, mesh::DGMultiMesh{3}, equations, dg::DGMultiSBP, cache;
                            normalize = true)

Compute the energy spectrum for a 3D `DGMulti` finite-difference SBP solution whose
nodes already form a uniform Cartesian grid.
"""
function compute_energy_spectrum(u, mesh::DGMultiMesh{3},
                                 equations::AbstractCompressibleEulerEquations,
                                 dg::DGMultiSBP, cache;
                                 normalize = true)
    primitive_variables = dgmulti_primitive_variables(u, equations, dg, Val(3))
    rho = primitive_variables[1]
    density_weighted_velocity_1 = sqrt.(rho) .* primitive_variables[2]
    density_weighted_velocity_2 = sqrt.(rho) .* primitive_variables[3]
    density_weighted_velocity_3 = sqrt.(rho) .* primitive_variables[4]

    return compute_energy_spectrum(density_weighted_velocity_1,
                                   density_weighted_velocity_2,
                                   density_weighted_velocity_3; normalize)
end
end # @muladd
