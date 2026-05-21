# By default, Julia/LLVM does not use fused multiply-add operations (FMAs)
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details
@muladd begin
#! format: noindent

"""
    compute_kinetic_energy_spectrum(v1, v2, v3)

Compute an isotropic 1D kinetic energy spectrum from three 3D Cartesian velocity
components `v1`, `v2`, `v3`. For compressible Euler kinetic energy spectra,
pass density-weighted components `sqrt(rho) * v1`, `sqrt(rho) * v2`, and `sqrt(rho) * v3`.
The modal energy is normalized by `1 / N^3`.
"""
function compute_kinetic_energy_spectrum(v1::AbstractArray{<:Any, 3},
                                         v2::AbstractArray{<:Any, 3},
                                         v3::AbstractArray{<:Any, 3})

    # Compute the energy modes using FFTW
    energy_modes = 0.5f0 .* (abs2.(fft(v1)) .+ abs2.(fft(v2)) .+ abs2.(fft(v3)))
    energy_modes ./= length(energy_modes)^2

    return radial_energy_spectrum(energy_modes)
end

"""
    compute_kinetic_energy_spectrum(u, mesh::TreeMesh{3}, equations, solver::DGSEM,
                                    cache)

Compute the energy spectrum for a non-AMR 3D `TreeMesh`/`DGSEM` solution by first
interpolating from LGL nodes to a uniform Cartesian grid.
"""
function compute_kinetic_energy_spectrum(u, mesh::TreeMesh{3},
                                         equations::AbstractCompressibleEulerEquations,
                                         solver::DGSEM, cache)
    # Interpolates conservative polynomials to a uniform Cartesian grid then converts to primitives at each uniform node
    u_uniform = interpolate_lgl_to_uniform_cartesian(u, mesh, equations, solver, cache)
    grid_size = size(u_uniform)[2:end] # the first dimension is the equation index so it is not needed to count the spatial indices
    rho = Array{eltype(u)}(undef, grid_size)
    v1 = Array{eltype(u)}(undef, grid_size)
    v2 = Array{eltype(u)}(undef, grid_size)
    v3 = Array{eltype(u)}(undef, grid_size)
    for idx in CartesianIndices(grid_size)
        u_node = get_node_vars(u_uniform, equations, solver, Tuple(idx)...)
        prim = cons2prim(u_node, equations)
        rho[idx] = prim[1]
        v1[idx] = prim[2]
        v2[idx] = prim[3]
        v3[idx] = prim[4]
    end
    # Converts primitive velocity components to density weighted form before FFT
    density_weighted_velocity_1 = sqrt.(rho) .* v1
    density_weighted_velocity_2 = sqrt.(rho) .* v2
    density_weighted_velocity_3 = sqrt.(rho) .* v3

    return compute_kinetic_energy_spectrum(density_weighted_velocity_1,
                                           density_weighted_velocity_2,
                                           density_weighted_velocity_3)
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

    # Uses one uniform interpolation node per DGSEM solution node in each coordinate
    # direction. A degree p element has p + 1 LGL nodes per direction, so this
    # is the minimum Cartesian sampling matching the element-wise geometry
    # but this can be increased to finer sampling sizes if needed
    n_uniform_nodes = polydeg(solver) + 1
    grid_points_per_dimension = n_uniform_nodes * cells_per_dimension

    n_vars = nvariables(equations)
    u_uniform = Array{eltype(u)}(undef, n_vars, grid_points_per_dimension,
                                 grid_points_per_dimension,
                                 grid_points_per_dimension)

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
        # Gather conservative nodal values on the reference LGL grid for the element
        u_sample = get_node_vars(u, equations, solver, 1, 1, 1, element)
        element_size = (n_vars, nnodes(solver), nnodes(solver),
                        nnodes(solver))
        element_values = Array{eltype(u_sample)}(undef, element_size)
        for k in eachnode(solver), j in eachnode(solver), i in eachnode(solver)
            u_node = get_node_vars(u, equations, solver, i, j, k, element)
            for variable in 1:n_vars
                element_values[variable, i, j, k] = u_node[variable]
            end
        end
        interpolated = multiply_dimensionwise(vandermonde, element_values)

        # Each element is placed on the uniform grid assuming reference directions align with
        # physical axes (ξ→x, η→y, ζ→z) and nodal values use the standard DGSEM tensor order along (ξ, η, ζ)
        first_index = Vector{Int}(undef, 3)
        for dim in 1:3
            lower_left = normalized_coordinates[dim, element] -
                         (n_uniform_nodes - 1) / 2 * dx_global
            first_index[dim] = round(Int,
                                     (lower_left - (-1 + dx_global / 2)) / dx_global) +
                               1
        end

        # Writes the interpolated block onto the global grid for the larger output
        # `r1`, `r2`, and `r3` are the global indices corresponding to `u_uniform` that this specific element's interpolated block fits within
        # Essentially, `r1`, `r2`, and `r3` are the positions of the current element within `u_uniform`
        # first_index[dim] refers to the first index of the nodes of this element within the array of global tensor nodes
        # See the sketch in `spectral_analysis_2d.jl` for a sketch in 2d of how the local to global assembly works
        r1 = first_index[1]:(first_index[1] + n_uniform_nodes - 1)
        r2 = first_index[2]:(first_index[2] + n_uniform_nodes - 1)
        r3 = first_index[3]:(first_index[3] + n_uniform_nodes - 1)
        u_uniform[:, r1, r2, r3] .= interpolated[:, :, :, :]
    end
    return u_uniform
end

"""
    compute_kinetic_energy_spectrum(u, mesh::DGMultiMesh{3}, equations,
                                    dg::DGMultiSBP, cache)

Compute the energy spectrum for a 3D `DGMulti` finite-difference SBP solution whose
nodes already form a uniform Cartesian grid.
"""
function compute_kinetic_energy_spectrum(u, mesh::DGMultiMesh{3},
                                         equations::AbstractCompressibleEulerEquations,
                                         dg::DGMultiSBP, cache)
    # Unpacks the primitive variables from the conservative state for FDSBP DGMulti solutions
    u_values = parent(u)
    n_points = length(u_values)
    n = round(Int, n_points^(1 / 3))
    q = cons2prim.(u_values, Ref(equations)) # q is the vector that contains the primitive variables for density and velocity converted from the conservative variables
    rho = reshape(getindex.(q, 1), n, n, n)
    density_weighted_velocity_1 = sqrt.(rho) .* reshape(getindex.(q, 2), n, n, n)
    density_weighted_velocity_2 = sqrt.(rho) .* reshape(getindex.(q, 3), n, n, n)
    density_weighted_velocity_3 = sqrt.(rho) .* reshape(getindex.(q, 4), n, n, n)

    return compute_kinetic_energy_spectrum(density_weighted_velocity_1,
                                           density_weighted_velocity_2,
                                           density_weighted_velocity_3)
end
end # @muladd
