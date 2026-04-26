# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

# Internal Cartesian grid FFT kernel that computes the energy spectrum; called from internal methods and power spectra methods
function _compute_energy_spectrum(velocity_cartesian::AbstractArray...;
                                  normalize = true)
    velocity_hat = map(fft, velocity_cartesian)
    energy_modes = zeros(real(eltype(first(velocity_hat))), size(first(velocity_hat)))
    for velocity_component_hat in velocity_hat
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
    maximum_wavenumber = floor(Int,
                               sqrt(sum((size(energy_modes, dim) ÷ 2)^2
                                        for dim in 1:ndims_energy_modes)) + 0.5)
    energy_spectrum = zeros(eltype(energy_modes), maximum_wavenumber + 1)
    wavenumbers = collect(0:maximum_wavenumber)

    for index in CartesianIndices(energy_modes)
        effective_wavenumber = sqrt(sum(begin
                                            wavenumber = index[dim] - 1
                                            if wavenumber > size(energy_modes, dim) ÷ 2
                                                wavenumber -= size(energy_modes, dim)
                                            end
                                            wavenumber^2
                                        end
                                        for dim in 1:ndims_energy_modes))
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
    data = interpolate_lgl_to_uniform_cartesian(u, mesh, equations, solver, cache)
    rho = data[1]
    velocities = ntuple(dim -> data[dim + 1], ndims(mesh))
    density_weighted_velocities = ntuple(dim -> sqrt.(rho) .* velocities[dim],
                                         length(velocities))
    return _compute_energy_spectrum(density_weighted_velocities...; normalize)
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

    # Select physically relevant fields by name
    u_values = u isa StructArray ? u : Base.parent(u)
    n_points = length(u_values)
    n_points_per_dimension = round(Int, n_points^(1 / NDIMS))
    if n_points_per_dimension^NDIMS != n_points
        throw(ArgumentError("DGMulti data does not form a uniform Cartesian grid"))
    end

    # Repack solution components into arrays by the number of points per dimension to handle direct Cartesian FFT
    data = [Array{real(dg)}(undef, ntuple(_ -> n_points_per_dimension, NDIMS))
            for _ in 1:(NDIMS + 1)]
    for index in eachindex(u_values)
        u_node = cons2prim(u_values[index], equations)
        for variable in eachindex(data)
            data[variable][index] = u_node[variable]
        end
    end

    rho = data[1]
    velocities = ntuple(dim -> data[dim + 1], NDIMS)
    density_weighted_velocities = ntuple(dim -> sqrt.(rho) .* velocities[dim],
                                         length(velocities))
    return _compute_energy_spectrum(density_weighted_velocities...; normalize)
end

# Interpolate DGSEM LGL data on uniform TreeMesh cells to a global Cartesian grid since LGL nodes are not equidistant for FFTs
function interpolate_lgl_to_uniform_cartesian(u, mesh::Union{TreeMesh{2}, TreeMesh{3}},
                                              equations::AbstractCompressibleEulerEquations,
                                              solver::DGSEM, cache)
    NDIMS = ndims(mesh)
    # Restrict to straightforward non AMR setups
    leaf_cell_ids = leaf_cells(mesh.tree)
    levels = mesh.tree.levels[leaf_cell_ids]
    if !all(==(first(levels)), levels)
        throw(ArgumentError("AMR meshes are not supported yet"))
    end

    level = first(levels)
    cells_per_dimension = 2^level
    if length(levels) != cells_per_dimension^NDIMS
        throw(ArgumentError("energy spectrum interpolation requires a complete " *
                            "uniform TreeMesh without missing leaf cells"))
    end
    nvisnodes = polydeg(solver) + 1

    data = [Array{real(solver)}(undef,
                                ntuple(_ -> nvisnodes * cells_per_dimension,
                                       NDIMS))
            for _ in 1:(NDIMS + 1)]

    # Interpolate from LGL nodes to cell centered equidistant nodes in each element
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
        # Gather local nodal values for all solution variables in this element
        first_node = ntuple(_ -> 1, Val(NDIMS))
        u_node = cons2prim(get_node_vars(u, equations, solver, first_node...,
                                         element),
                           equations)
        local_data = Array{eltype(u_node)}(undef, length(data),
                                           ntuple(_ -> nnodes(solver), NDIMS)...)
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

        first_index = ntuple(dim -> begin
                                 lower_left = normalized_coordinates[dim, element] -
                                              (nvisnodes - 1) / 2 * dx_global
                                 round(Int,
                                       (lower_left - (-1 + dx_global / 2)) /
                                       dx_global) + 1
                             end,
                             Val(NDIMS))
        element_indices = ntuple(dim -> first_index[dim]:(first_index[dim] + nvisnodes - 1),
                                 Val(NDIMS))
        for variable in eachindex(data)
            data[variable][element_indices...] .= selectdim(interpolated, 1, variable)
        end
    end

    return data
end
end # @muladd
