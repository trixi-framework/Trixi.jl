# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

"""
    compute_energy_spectrum(rho_cartesian, velocity_cartesian...; normalize = true)
    compute_energy_spectrum(sol; solution_variables = nothing, nvisnodes = nnodes(solver))

Computes the radially averaged kinetic energy spectrum of a numerical solution sampled on a
uniform Cartesian grid.

The first method expects density and velocity arrays that have already been sampled on a uniform
Cartesian grid. The second method extracts these arrays from a Trixi solution. For standard
`TreeMesh`/`DGSEM` discretizations, the solution is interpolated from Legendre-Gauss-Lobatto nodes
to a uniform Cartesian grid. For `DGMulti`/finite-difference SBP discretizations, the solution is
already represented on a uniform Cartesian grid and no interpolation is applied.

If `normalize` is `true`, the spectrum is normalized consistently with Julia's unnormalized FFT
such that `sum(energy_spectrum)` is equal to the mean kinetic energy on the Cartesian grid.

Returns `(energy_spectrum, wavenumbers)` where `energy_spectrum[i]` contains the energy in the
shell centered at integer wavenumber `wavenumbers[i]`.
"""
function compute_energy_spectrum(rho_cartesian::AbstractArray,
                                 velocity_cartesian::AbstractArray...;
                                 normalize = true)
    # FFT indexing and Cartesian shell binning below assume standard Julia indexing.
    Base.require_one_based_indexing(rho_cartesian, velocity_cartesian...)

    ndims_cartesian = ndims(rho_cartesian)
    if ndims_cartesian != 2 && ndims_cartesian != 3
        throw(ArgumentError("`rho_cartesian` must be a 2D or 3D array sampled " *
                            "on a uniform Cartesian grid"))
    end

    if length(velocity_cartesian) != ndims_cartesian
        throw(ArgumentError("expected $ndims_cartesian velocity components for " *
                            "$(ndims_cartesian)D input, got $(length(velocity_cartesian))"))
    end

    for velocity in velocity_cartesian
        if size(velocity) != size(rho_cartesian)
            throw(ArgumentError("all velocity component arrays must have the same size " *
                                "as `rho_cartesian`"))
        end
    end

    # Calculates density weighted velocities in Fourier space using FFTW
    density_weighted_velocity_hat = map(velocity -> fft(sqrt.(rho_cartesian) .*
                                                        velocity),
                                        velocity_cartesian)
    energy_modes = zero(0.5 .* abs2.(first(density_weighted_velocity_hat)))
    for velocity_hat in density_weighted_velocity_hat
        energy_modes .+= 0.5 .* abs2.(velocity_hat)
    end

    # Accounts for unnormalized FFT convention if specified
    if normalize
        energy_modes ./= length(energy_modes)^2
    end

    # Bin Cartesian Fourier modes into isotropic shells
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

# Entry point for ODE solutions
function compute_energy_spectrum(sol; kwargs...)
    return compute_energy_spectrum(sol.u[end], sol.prob.p; kwargs...)
end

# Entry point for raw state vectors together with semidiscretization
function compute_energy_spectrum(u_ode, semi::AbstractSemidiscretization; kwargs...)
    return compute_energy_spectrum(wrap_array_native(u_ode, semi),
                                   mesh_equations_solver_cache(semi)...; kwargs...)
end

# Specifcally for 2D and 3D Treemeshes that interpolate from the LGL nodes to a uniform Cartesian grid
function compute_energy_spectrum(u, mesh::Union{TreeMesh{2}, TreeMesh{3}}, equations,
                                 solver::DGSEM, cache;
                                 solution_variables = nothing,
                                 nvisnodes = nnodes(solver),
                                 density_variable = "rho", velocity_variables = nothing,
                                 normalize = true)
    data, variable_names = interpolate_lgl_to_uniform_cartesian(u, mesh, equations,
                                                                solver,
                                                                cache;
                                                                solution_variables,
                                                                nvisnodes)

    # Checks for fields by name since the current scope does not neccasiraly gaurentee correctness, for example if the user passes a custom solution_variables
    density_id = findfirst(isequal(String(density_variable)), variable_names)
    if isnothing(density_id)
        throw(ArgumentError("could not find variable `$density_variable` in solution variables"))
    end

    velocity_variables_ = velocity_variables === nothing ?
                          ntuple(dim -> "v$dim", ndims(first(data))) :
                          velocity_variables
    velocity_ids = map(variable -> begin
                           variable_id = findfirst(isequal(String(variable)),
                                                   variable_names)
                           if isnothing(variable_id)
                               throw(ArgumentError("could not find variable `$variable` in solution variables"))
                           end
                           variable_id
                       end, velocity_variables_)

    rho = data[density_id]
    velocities = ntuple(dim -> data[velocity_ids[dim]], length(velocity_ids))
    return compute_energy_spectrum(rho, velocities...; normalize)
end

# DGMulti path used for  uniform Cartesian discretizations that dont need to be interpolated
function compute_energy_spectrum(u, mesh::DGMultiMesh{NDIMS}, equations, dg::DGMulti,
                                 cache;
                                 solution_variables = nothing,
                                 density_variable = "rho", velocity_variables = nothing,
                                 normalize = true) where {NDIMS}
    if NDIMS != 2 && NDIMS != 3
        throw(ArgumentError("only 2D and 3D DGMulti discretizations are supported"))
    end
    if !(dg.basis.approximation_type isa AbstractPeriodicDerivativeOperator)
        throw(ArgumentError("only DGMulti finite-difference SBP data on uniform " *
                            "Cartesian grids is supported"))
    end

    # Select physically relevant fields by name
    solution_variables_ = if isnothing(solution_variables)
        if hasmethod(cons2prim, Tuple{AbstractVector, typeof(equations)})
            cons2prim
        else
            cons2cons
        end
    else
        solution_variables
    end
    variable_names = SVector(varnames(solution_variables_, equations))
    u_values = u isa StructArray ? u : Base.parent(u)
    n_points = length(u_values)
    n_points_per_dimension = round(Int, n_points^(1 / NDIMS))
    if n_points_per_dimension^NDIMS != n_points
        throw(ArgumentError("DGMulti data does not form a uniform Cartesian grid"))
    end

    # Repack solution components into arrays by the number of points per dimension to handle direct Cartesian FFT
    data = [Array{real(dg)}(undef, ntuple(_ -> n_points_per_dimension, NDIMS))
            for _ in eachindex(variable_names)]
    for index in eachindex(u_values)
        u_node = solution_variables_(u_values[index], equations)
        for variable in eachindex(variable_names)
            data[variable][index] = u_node[variable]
        end
    end

    # Still checks for fields by name since the current scope does not neccasiraly gaurentee correctness
    density_id = findfirst(isequal(String(density_variable)), variable_names)
    if isnothing(density_id)
        throw(ArgumentError("could not find variable `$density_variable` in solution variables"))
    end
    velocity_variables_ = velocity_variables === nothing ?
                          ntuple(dim -> "v$dim", ndims(first(data))) :
                          velocity_variables
    velocity_ids = map(variable -> begin
                           variable_id = findfirst(isequal(String(variable)),
                                                   variable_names)
                           if isnothing(variable_id)
                               throw(ArgumentError("could not find variable `$variable` in solution variables"))
                           end
                           variable_id
                       end, velocity_variables_)

    rho = data[density_id]
    velocities = ntuple(dim -> data[velocity_ids[dim]], length(velocity_ids))
    return compute_energy_spectrum(rho, velocities...; normalize)
end

# Interpolate DGSEM LGL data on uniform TreeMesh cells to a global Cartesian grid since LGL nodes are not equidistant for FFTs
function interpolate_lgl_to_uniform_cartesian(u, mesh::TreeMesh{NDIMS}, equations,
                                              solver::DGSEM, cache;
                                              solution_variables = nothing,
                                              nvisnodes = nnodes(solver)) where {NDIMS}
    # Restrict to straightforward non AMR setups
    if !(nvisnodes isa Integer && nvisnodes >= 2)
        throw(ArgumentError("`nvisnodes` must be an integer >= 2"))
    end
    leaf_cell_ids = leaf_cells(mesh.tree)
    levels = mesh.tree.levels[leaf_cell_ids]
    if !all(==(first(levels)), levels)
        throw(ArgumentError("AMR meshes are not supported yet"))
    end

    level = first(levels)
    cells_per_dimension = 2^level
    if length(levels) != cells_per_dimension^NDIMS
        throw(ArgumentError("only complete uniform TreeMesh discretizations are " *
                            "supported"))
    end

    solution_variables_ = if isnothing(solution_variables)
        if hasmethod(cons2prim, Tuple{AbstractVector, typeof(equations)})
            cons2prim
        else
            cons2cons
        end
    else
        solution_variables
    end
    variable_names = SVector(varnames(solution_variables_, equations))
    data = [Array{real(solver)}(undef,
                                ntuple(_ -> nvisnodes * cells_per_dimension,
                                       NDIMS))
            for _ in eachindex(variable_names)]

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
        u_node = solution_variables_(get_node_vars(u, equations, solver, first_node...,
                                                   element),
                                     equations)
        local_data = Array{eltype(u_node)}(undef, length(variable_names),
                                           ntuple(_ -> nnodes(solver), NDIMS)...)
        for node in CartesianIndices(Base.tail(size(local_data)))
            u_node = solution_variables_(get_node_vars(u, equations, solver,
                                                       Tuple(node)..., element),
                                         equations)
            for variable in eachindex(variable_names)
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
        for variable in eachindex(variable_names)
            data[variable][element_indices...] .= selectdim(interpolated, 1, variable)
        end
    end

    return data, variable_names
end
end # @muladd
