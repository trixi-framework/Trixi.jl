# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

"""
    CartesianSolutionData(coordinates, data, variable_names)

Container for solution data sampled on a uniform Cartesian grid.
"""
struct CartesianSolutionData{NDIMS, Coordinates, Data, VariableNames}
    coordinates::Coordinates
    data::Data
    variable_names::VariableNames
end

function Base.getindex(cartesian::CartesianSolutionData, variable_name)
    variable_id = findfirst(isequal(String(variable_name)), cartesian.variable_names)

    if isnothing(variable_id)
        throw(KeyError(variable_name))
    end

    return cartesian.data[variable_id]
end

"""
    compute_energy_spectrum(rho_cartesian, velocity_cartesian...; normalize = true)

Computes the radially averaged kinetic energy spectrum of a numerical solution sampled on a
uniform Cartesian grid.

The input arrays must be sampled on a uniform Cartesian grid and must have identical sizes.
Two-dimensional input requires two velocity components, and three-dimensional input requires three
velocity components. The Fourier transform is applied to the density-weighted velocities
`sqrt(rho_cartesian) .* velocity_cartesian[i]`.

If `normalize` is `true`, the spectrum is normalized consistently with Julia's unnormalized FFT
such that `sum(energy_spectrum)` is equal to the mean kinetic energy on the Cartesian grid.

Returns `(energy_spectrum, wavenumbers)` where `energy_spectrum[i]` contains the energy in the
shell centered at integer wavenumber `wavenumbers[i]`.
"""
function compute_energy_spectrum(rho_cartesian::AbstractArray,
                                 velocity_cartesian::AbstractArray...;
                                 normalize = true)
    Base.require_one_based_indexing(rho_cartesian, velocity_cartesian...)

    ndims_cartesian = ndims(rho_cartesian)
    if ndims_cartesian != 2 && ndims_cartesian != 3
        throw(ArgumentError("`rho_cartesian` must be a 2D or 3D array sampled on a uniform Cartesian grid"))
    end

    if length(velocity_cartesian) != ndims_cartesian
        throw(ArgumentError("expected $ndims_cartesian velocity components for $(ndims_cartesian)D input, got $(length(velocity_cartesian))"))
    end

    for velocity in velocity_cartesian
        if size(velocity) != size(rho_cartesian)
            throw(ArgumentError("all velocity component arrays must have the same size as `rho_cartesian`"))
        end
    end

    density_weighted_velocity_hat = map(velocity -> fft(sqrt.(rho_cartesian) .*
                                                        velocity), velocity_cartesian)

    energy_modes = 0.5 .* abs2.(first(density_weighted_velocity_hat))
    for velocity_hat in Iterators.drop(density_weighted_velocity_hat, 1)
        energy_modes .+= 0.5 .* abs2.(velocity_hat)
    end

    if normalize
        energy_modes ./= length(energy_modes)^2
    end

    return radial_energy_spectrum(energy_modes)
end

"""
    interpolate_to_uniform_cartesian(sol; solution_variables = nothing, nvisnodes = nnodes(solver))
    interpolate_to_uniform_cartesian(u_ode, semi; solution_variables = nothing, nvisnodes = nnodes(solver))

Interpolate a standard non-AMR `TreeMesh`/`DGSEM` solution from Legendre-Gauss-Lobatto nodes to a
uniform Cartesian grid.

This helper currently supports two- and three-dimensional non-AMR `TreeMesh` discretizations using
`DGSEM`. The returned [`CartesianSolutionData`](@ref) can be used as input for
[`compute_energy_spectrum`](@ref).
"""
function interpolate_to_uniform_cartesian(sol; kwargs...)
    return interpolate_to_uniform_cartesian(sol.u[end], sol.prob.p; kwargs...)
end

function interpolate_to_uniform_cartesian(u_ode, semi::AbstractSemidiscretization;
                                          kwargs...)
    return interpolate_to_uniform_cartesian(wrap_array_native(u_ode, semi),
                                            mesh_equations_solver_cache(semi)...;
                                            kwargs...)
end

function interpolate_to_uniform_cartesian(u, mesh::TreeMesh, equations,
                                          solver::DGSEM, cache;
                                          solution_variables = nothing,
                                          nvisnodes = nnodes(solver))
    ndims_mesh = ndims(mesh)
    if ndims_mesh != 2 && ndims_mesh != 3
        throw(ArgumentError("only 2D and 3D TreeMesh/DGSEM discretizations are supported"))
    end

    levels = mesh.tree.levels[leaf_cells(mesh.tree)]
    if !all(==(first(levels)), levels)
        throw(ArgumentError("AMR meshes are not supported yet"))
    end

    if !(nvisnodes isa Integer && nvisnodes >= 2)
        throw(ArgumentError("`nvisnodes` must be an integer >= 2"))
    end

    solution_variables_ = postprocess_solution_variables(equations, solution_variables)
    variable_names = SVector(varnames(solution_variables_, equations))

    level = first(levels)
    cells_per_dimension = 2^level
    if length(levels) != cells_per_dimension^ndims_mesh
        throw(ArgumentError("only complete uniform TreeMesh discretizations are supported"))
    end

    resolution = nvisnodes * cells_per_dimension

    coordinates = uniform_cartesian_coordinates(mesh, resolution)
    data = [Array{real(solver)}(undef, ntuple(_ -> resolution, ndims_mesh))
            for _ in eachindex(variable_names)]

    nodes_out = uniform_reference_nodes(nvisnodes)
    vandermonde = polynomial_interpolation_matrix(get_nodes(solver.basis), nodes_out)
    leaf_cell_ids = leaf_cells(mesh.tree)
    normalized_coordinates = normalized_leaf_cell_coordinates(mesh, leaf_cell_ids)

    for element in eachelement(solver, cache)
        local_data = local_solution_variables(u, equations, solver, element,
                                              solution_variables_, ndims_mesh,
                                              length(variable_names))
        interpolated_data = multiply_dimensionwise(vandermonde, local_data)
        first_index = first_cartesian_index(normalized_coordinates, element,
                                            resolution, nvisnodes)
        element_indices = ntuple(dim -> first_index[dim]:(first_index[dim] + nvisnodes - 1),
                                 Val(ndims_mesh))

        for variable in eachindex(variable_names)
            data[variable][element_indices...] .= selectdim(interpolated_data, 1,
                                                            variable)
        end
    end

    return CartesianSolutionData{ndims_mesh, typeof(coordinates), typeof(data),
                                 typeof(variable_names)}(coordinates, data,
                                                         variable_names)
end

function postprocess_solution_variables(equations, solution_variables)
    return solution_variables
end

function postprocess_solution_variables(equations, solution_variables::Nothing)
    if hasmethod(cons2prim, Tuple{AbstractVector, typeof(equations)})
        return cons2prim
    else
        return cons2cons
    end
end

function uniform_reference_nodes(nvisnodes)
    dx = 2 / nvisnodes
    return collect(range(-1 + dx / 2, 1 - dx / 2, length = nvisnodes))
end

function uniform_cartesian_coordinates(mesh::TreeMesh, resolution)
    center = mesh.tree.center_level_0
    length = mesh.tree.length_level_0
    dx = length / resolution

    return ntuple(dim -> collect(range(center[dim] - length / 2 + dx / 2,
                                       center[dim] + length / 2 - dx / 2,
                                       length = resolution)),
                  Val(ndims(mesh)))
end

function normalized_leaf_cell_coordinates(mesh::TreeMesh, leaf_cell_ids)
    center = mesh.tree.center_level_0
    length = mesh.tree.length_level_0
    coordinates = mesh.tree.coordinates[:, leaf_cell_ids]
    normalized_coordinates = similar(coordinates)

    for element in axes(coordinates, 2)
        @views normalized_coordinates[:, element] .= ((coordinates[:, element] .-
                                                       center) ./
                                                      (length / 2))
    end

    return normalized_coordinates
end

function local_solution_variables(u, equations, solver, element, solution_variables,
                                  ndims_mesh, n_variables)
    first_node = ntuple(_ -> 1, Val(ndims_mesh))
    u_node = solution_variables(get_node_vars(u, equations, solver, first_node...,
                                              element),
                                equations)
    local_data = Array{eltype(u_node)}(undef, n_variables,
                                       ntuple(_ -> nnodes(solver), ndims_mesh)...)

    for node in CartesianIndices(Base.tail(size(local_data)))
        u_node = solution_variables(get_node_vars(u, equations, solver,
                                                  Tuple(node)..., element),
                                    equations)
        for variable in 1:n_variables
            local_data[variable, Tuple(node)...] = u_node[variable]
        end
    end

    return local_data
end

function first_cartesian_index(normalized_coordinates, element, resolution, nvisnodes)
    ndims_mesh = size(normalized_coordinates, 1)
    dx = 2 / resolution

    return ntuple(dim -> begin
                      lower_left = normalized_coordinates[dim, element] -
                                   (nvisnodes - 1) / 2 * dx
                      round(Int, (lower_left - (-1 + dx / 2)) / dx) + 1
                  end,
                  Val(ndims_mesh))
end

# Sum Fourier modes into shells centered at integer wavenumbers.
function radial_energy_spectrum(energy_modes)
    ndims_energy_modes = ndims(energy_modes)
    maximum_wavenumber = maximum_shell(size(energy_modes))

    energy_spectrum = zeros(eltype(energy_modes), maximum_wavenumber + 1)
    wavenumbers = collect(0:maximum_wavenumber)

    for index in CartesianIndices(energy_modes)
        effective_wavenumber = sqrt(sum(wavenumber_at_index(index[dim],
                                                            size(energy_modes, dim))^2
                                        for dim in 1:ndims_energy_modes))
        shell = floor(Int, effective_wavenumber + 0.5)

        if shell <= maximum_wavenumber
            energy_spectrum[shell + 1] += energy_modes[index]
        end
    end
    return energy_spectrum, wavenumbers
end

function maximum_shell(size_energy_modes)
    maximum_wavenumber = sqrt(sum((size_energy_modes[dim] ÷ 2)^2
                                  for dim in eachindex(size_energy_modes)))
    return floor(Int, maximum_wavenumber + 0.5)
end

function wavenumber_at_index(index, n_indices)
    wavenumber = index - 1
    if wavenumber > n_indices ÷ 2
        wavenumber -= n_indices
    end

    return wavenumber
end
end # @muladd
