# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

# Internal Cartesian grid FFT kernel that computes the energy spectrum
function compute_energy_spectrum(velocity_cartesian...;
                                  normalize = true)

    # Handles the case where the user passes a tuple of velocity components
    if length(velocity_cartesian) == 1 && first(velocity_cartesian) isa Tuple
        velocity_cartesian = first(velocity_cartesian)
    end

    # `velocity_cartesian` is expected to contain one array per spatial velocity component,
    # ordered as `(v1, v2)` in 2D or `(v1, v2, v3)` in 3D. Each array must contain values
    # sampled on the same uniform Cartesian grid. For compressible Euler spectra, callers pass
    # density-weighted velocities `sqrt(rho) * v_i`
    first_velocity_hat = fft(first(velocity_cartesian))
    energy_modes = 0.5 .* abs2.(first_velocity_hat)

    for component_id in 2:length(velocity_cartesian)
        velocity_component_hat = fft(velocity_cartesian[component_id])
        energy_modes .+= 0.5 .* abs2.(velocity_component_hat)
    end

    # Accounts for unnormalized FFT convention if specified
    if normalize
        energy_modes ./= length(energy_modes)^2
    end

    # Convert the multi-dimensional Fourier energy into the
    # 1d isotropic spectrum E by summing all modes whose
    # wavenumber rounds to the same integer shell, accounting for the FFT convention
    ndims_energy_modes = ndims(energy_modes)
    maximum_wavenumber_squared = 0
    for dim in 1:ndims_energy_modes
        maximum_wavenumber_squared += (size(energy_modes, dim) ÷ 2)^2
    end
    maximum_wavenumber = floor(Int, sqrt(maximum_wavenumber_squared) + 0.5)
    energy_spectrum = zeros(eltype(energy_modes), maximum_wavenumber + 1)
    wavenumbers = collect(0:maximum_wavenumber)

    for index in CartesianIndices(energy_modes)
        effective_wavenumber_squared = 0
        for dim in 1:ndims_energy_modes
            wavenumber = index[dim] - 1
            if wavenumber > size(energy_modes, dim) ÷ 2
                wavenumber -= size(energy_modes, dim)
            end
            effective_wavenumber_squared += wavenumber^2
        end

        effective_wavenumber = sqrt(effective_wavenumber_squared)
        shell = floor(Int, effective_wavenumber + 0.5)
        if shell <= maximum_wavenumber
            energy_spectrum[shell + 1] += energy_modes[index]
        end
    end

    return energy_spectrum, wavenumbers
end

"""
    compute_energy_spectrum(sol; kwargs...)

Compute the energy spectrum from the final state of an ODE solution returned by
`solve`. Keyword arguments are forwarded to the semidiscretization-specific method.

For density-weighted kinetic energy spectra in compressible turbulence, see Winters AR,
Moura RC, Mengaldo G, Gassner GJ, Walch S, Peiro J, et al. A comparative study on
polynomial dealiasing and split form discontinuous Galerkin schemes for under-resolved
turbulence computations. Journal of Computational Physics 372 (2018), 1-21.
"""
function compute_energy_spectrum(sol; kwargs...)
    return compute_energy_spectrum(sol.u[end], sol.prob.p; kwargs...)
end

"""
    compute_energy_spectrum(u_ode, semi; kwargs...)

Compute the energy spectrum from an ODE state vector `u_ode` and a Trixi
semidiscretization `semi`. The state is converted to the solver-native array
layout before dispatching to the mesh/solver-specific method.
"""
function compute_energy_spectrum(u_ode, semi::AbstractSemidiscretization; kwargs...)
    return compute_energy_spectrum(wrap_array_native(u_ode, semi),
                                   mesh_equations_solver_cache(semi)...; kwargs...)
end

"""
    compute_energy_spectrum(u, mesh::Union{TreeMesh{2}, TreeMesh{3}}, equations,
                            solver::DGSEM, cache; normalize = true)

Compute the energy spectrum for a non-AMR `TreeMesh`/`DGSEM` solution by first
interpolating the solution from LGL nodes to a uniform Cartesian grid.
"""
function compute_energy_spectrum(u, mesh::Union{TreeMesh{2}, TreeMesh{3}},
                                 equations::AbstractCompressibleEulerEquations,
                                 solver::DGSEM, cache;
                                 normalize = true)
    leaf_cell_ids = leaf_cells(mesh.tree)
    levels = mesh.tree.levels[leaf_cell_ids]
    if !all(==(first(levels)), levels)
        throw(ArgumentError("AMR meshes are not supported yet"))
    end

    NDIMS = ndims(mesh)
    data = interpolate_lgl_to_uniform_cartesian(u, mesh, equations, solver, cache)
    rho = data[1]

    if NDIMS == 2
        density_weighted_velocity_1 = sqrt.(rho) .* data[2]
        density_weighted_velocity_2 = sqrt.(rho) .* data[3]
        return _compute_energy_spectrum(density_weighted_velocity_1,
                                        density_weighted_velocity_2; normalize)
    else
        density_weighted_velocity_1 = sqrt.(rho) .* data[2]
        density_weighted_velocity_2 = sqrt.(rho) .* data[3]
        density_weighted_velocity_3 = sqrt.(rho) .* data[4]
        return _compute_energy_spectrum(density_weighted_velocity_1,
                                        density_weighted_velocity_2,
                                        density_weighted_velocity_3; normalize)
    end
end

"""
    compute_energy_spectrum(u, mesh::DGMultiMesh{NDIMS}, equations, dg::DGMulti, cache;
                            normalize = true)

Compute the energy spectrum for a `DGMulti` finite-difference SBP solution whose
nodes already form a uniform Cartesian grid.
"""
function compute_energy_spectrum(u, mesh::Union{DGMultiMesh{2}, DGMultiMesh{3}},
                                 equations::AbstractCompressibleEulerEquations,
                                 dg::DGMultiSBP,
                                 cache;
                                 normalize = true)
    NDIMS = ndims(mesh)

    # Uses primitive variables from the solution vector
    u_values = u isa StructArray ? u : Base.parent(u)
    n_points = length(u_values)
    n_points_per_dimension = round(Int, n_points^(1 / NDIMS))
    if n_points_per_dimension^NDIMS != n_points
        throw(ArgumentError("DGMulti data does not form a uniform Cartesian grid"))
    end

    # Repack solution components into Cartesian arrays for the FFT kernel.
    # `data` stores primitive variables as `[rho, v1, v2]` in 2D or
    # `[rho, v1, v2, v3]` in 3D. Each entry has size
    # `(n_points_per_dimension, n_points_per_dimension)` in 2D or
    # `(n_points_per_dimension, n_points_per_dimension, n_points_per_dimension)` in 3D
    data_size = if NDIMS == 2
        (n_points_per_dimension, n_points_per_dimension)
    else
        (n_points_per_dimension, n_points_per_dimension, n_points_per_dimension)
    end

    data = Vector{Array{real(dg), NDIMS}}(undef, NDIMS + 1)
    for variable in eachindex(data)
        data[variable] = Array{real(dg)}(undef, data_size)
    end

    for index in eachindex(u_values)
        u_node = cons2prim(u_values[index], equations)
        for variable in eachindex(data)
            data[variable][index] = u_node[variable]
        end
    end

    rho = data[1]
    if NDIMS == 2
        density_weighted_velocity_1 = sqrt.(rho) .* data[2]
        density_weighted_velocity_2 = sqrt.(rho) .* data[3]
        return _compute_energy_spectrum(density_weighted_velocity_1,
                                        density_weighted_velocity_2; normalize)
    else
        density_weighted_velocity_1 = sqrt.(rho) .* data[2]
        density_weighted_velocity_2 = sqrt.(rho) .* data[3]
        density_weighted_velocity_3 = sqrt.(rho) .* data[4]
        return _compute_energy_spectrum(density_weighted_velocity_1,
                                        density_weighted_velocity_2,
                                        density_weighted_velocity_3; normalize)
    end
end

# Interpolate DGSEM LGL data on uniform TreeMesh cells to a global Cartesian grid
function interpolate_lgl_to_uniform_cartesian(u, mesh::Union{TreeMesh{2}, TreeMesh{3}},
                                              equations::AbstractCompressibleEulerEquations,
                                              solver::DGSEM, cache)
    # Restrict to straightforward non AMR setups
    leaf_cell_ids = leaf_cells(mesh.tree)
    levels = mesh.tree.levels[leaf_cell_ids]
    if !all(==(first(levels)), levels)
        throw(ArgumentError("AMR meshes are not supported yet"))
    end

    NDIMS = ndims(mesh)
    level = first(levels)
    cells_per_dimension = 2^level

    nvisnodes = polydeg(solver) + 1
    grid_points_per_dimension = nvisnodes * cells_per_dimension

    # `data` stores primitive variables as `[rho, v1, v2]` in 2D or
    # `[rho, v1, v2, v3]` in 3D after interpolation to the global Cartesian grid
    # Each entry has `grid_points_per_dimension` points in every coordinate direction
    data_size = if NDIMS == 2
        (grid_points_per_dimension, grid_points_per_dimension)
    else
        (grid_points_per_dimension, grid_points_per_dimension,
         grid_points_per_dimension)
    end

    data = Vector{Array{real(solver), NDIMS}}(undef, NDIMS + 1)
    for variable in eachindex(data)
        data[variable] = Array{real(solver)}(undef, data_size)
    end

    # Interpolate from LGL nodes to cell-centered equidistant nodes in each element
    dx_reference = 2 / nvisnodes
    nodes_out = collect(range(-1 + dx_reference / 2, 1 - dx_reference / 2,
                              length = nvisnodes))
    vandermonde = polynomial_interpolation_matrix(get_nodes(solver.basis), nodes_out)

    # Element center coordinates determine where each interpolated block sits in the global grid
    center = reshape(mesh.tree.center_level_0, :, 1)
    normalized_coordinates = (mesh.tree.coordinates[:, leaf_cell_ids] .- center) ./
                             (mesh.tree.length_level_0 / 2)
    dx_global = 2 / (nvisnodes * cells_per_dimension)

    for element in eachelement(solver, cache)
        # Gather local nodal values for all primitive variables in this element
        first_node = ntuple(_ -> 1, Val(NDIMS))
        u_node = cons2prim(get_node_vars(u, equations, solver, first_node...,
                                         element),
                           equations)
        local_data_size = if NDIMS == 2
            (length(data), nnodes(solver), nnodes(solver))
        else
            (length(data), nnodes(solver), nnodes(solver), nnodes(solver))
        end
        local_data = Array{eltype(u_node)}(undef, local_data_size)
        for node in CartesianIndices(Base.tail(size(local_data)))
            u_node = cons2prim(get_node_vars(u, equations, solver,
                                             Tuple(node)..., element),
                               equations)
            for variable in eachindex(data)
                local_data[variable, Tuple(node)...] = u_node[variable]
            end
        end

        # Interpolate in each dimension using the tensor product structure
        interpolated = multiply_dimensionwise(vandermonde, local_data)

        first_index = Vector{Int}(undef, NDIMS)
        for dim in 1:NDIMS
            lower_left = normalized_coordinates[dim, element] -
                         (nvisnodes - 1) / 2 * dx_global
            first_index[dim] = round(Int,
                                     (lower_left - (-1 + dx_global / 2)) /
                                     dx_global) + 1
        end

        element_indices = if NDIMS == 2
            (first_index[1]:(first_index[1] + nvisnodes - 1),
             first_index[2]:(first_index[2] + nvisnodes - 1))
        else
            (first_index[1]:(first_index[1] + nvisnodes - 1),
             first_index[2]:(first_index[2] + nvisnodes - 1),
             first_index[3]:(first_index[3] + nvisnodes - 1))
        end
        for variable in eachindex(data)
            data[variable][element_indices...] .= selectdim(interpolated, 1, variable)
        end
    end

    return data
end
end # @muladd
