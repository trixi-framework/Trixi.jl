# By default, Julia/LLVM does not use fused multiply-add operations (FMAs)
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details
@muladd begin
#! format: noindent

"""
    compute_kinetic_energy_spectrum(v1, v2)

Compute an isotropic 1D kinetic energy spectrum from two 2D Cartesian velocity
components `v1` and `v2`. For compressible Euler kinetic energy spectra,
pass density-weighted components `sqrt(rho) * v1` and `sqrt(rho) * v2`.
The modal energy is normalized by `1 / N^2'.
"""
function compute_kinetic_energy_spectrum(v1::AbstractArray{<:Any, 2},
                                         v2::AbstractArray{<:Any, 2})

    # Compute the energy modes using FFTW
    energy_modes = 0.5f0 .* (abs2.(fft(v1)) .+ abs2.(fft(v2)))
    energy_modes ./= length(energy_modes)^2

    return radial_energy_spectrum(energy_modes)
end

"""
    compute_kinetic_energy_spectrum(u, mesh::TreeMesh{2}, equations, solver::DGSEM,
                                    cache)

Compute the energy spectrum for a non-AMR 2D `TreeMesh`/`DGSEM` solution by first
interpolating from LGL nodes to a uniform Cartesian grid.
"""
function compute_kinetic_energy_spectrum(u, mesh::TreeMesh{2},
                                         equations::AbstractCompressibleEulerEquations,
                                         solver::DGSEM, cache)
    # Interpolates conservative polynomials to a uniform Cartesian grid then converts to primitives at each uniform node
    u_uniform = interpolate_lgl_to_uniform_cartesian(u, mesh, equations, solver, cache)
    grid_size = size(u_uniform)[2:end] # the first dimension is the equation index so it is not needed to count the spatial indices
    rho = Array{eltype(u)}(undef, grid_size)
    v1 = Array{eltype(u)}(undef, grid_size)
    v2 = Array{eltype(u)}(undef, grid_size)
    for idx in CartesianIndices(grid_size)
        u_node = get_node_vars(u_uniform, equations, solver, Tuple(idx)...)
        prim = cons2prim(u_node, equations)
        rho[idx] = prim[1]
        v1[idx] = prim[2]
        v2[idx] = prim[3]
    end
    # Convert primitive velocity components to density weighted form before FFT
    density_weighted_velocity_1 = sqrt.(rho) .* v1
    density_weighted_velocity_2 = sqrt.(rho) .* v2

    return compute_kinetic_energy_spectrum(density_weighted_velocity_1,
                                           density_weighted_velocity_2)
end

function interpolate_lgl_to_uniform_cartesian(u, mesh::TreeMesh{2},
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

    # Example for 1st order (2 nodes per direction per element) for a 2x2 tree of cells which produces a 4x4 global FFT grid.
    #
    # Reference element — local indices (i along ξ, j along η):
    #
    #      j=2  (1,2)--------(2,2)
    #      ↑     |             |
    #      |     |             |
    #      |     |             |
    #      |     |             |
    #      j    (1,1)--------(2,1)
    #             ---------> i (ξ)
    #
    #   Each of the 4 tree cells pastes a 2×2 block into the 4×4 array global grid, 'u_uniform'.
    #   Labels (column, row) are indices into that array: column ↔ x, row ↔ y on the domain.
    #   The assignment `u_uniform[:, r1, r2] .= interpolated` fills one such rectangle.
    #   The bottom-right element of the original 2×2 block is labeled as 'E' in the diagram below and fills
    #   rows 3 to 4 and columns 3 to 4 of the global grid.
    #
    #         column  1         2         3         4
    #                 +---------+---------+---------+---------+
    #         row 1   | (1,1)   | (2,1)   | (3,1)   | (4,1)   |
    #                 +---------+---------+---------+---------+
    #         row 2   | (1,2)   | (2,2)   | (3,2)   | (4,2)   |
    #                 +---------+---------+---------+---------+   ↑
    #         row 3   | (1,3)   | (2,3)   | (3,3) E | (4,3) E |   |
    #                 +---------+---------+---------+---------+   |  r2 = 3:4
    #         row 4   | (1,4)   | (2,4)   | (3,4) E | (4,4) E |   |
    #                 +---------+---------+---------+---------+   ↓
    #                                     |<-----r1 = 3:4---->|
    #

    for element in eachelement(solver, cache)
        # Gather conservative nodal values on the reference LGL tensor grid for the element
        u_sample = get_node_vars(u, equations, solver, 1, 1, element)
        element_size = (n_vars, nnodes(solver), nnodes(solver))
        element_values = Array{eltype(u_sample)}(undef,
                                                 element_size)
        for j in eachnode(solver), i in eachnode(solver)
            u_node = get_node_vars(u, equations, solver, i, j, element)
            for variable in 1:n_vars
                element_values[variable, i, j] = u_node[variable]
            end
        end
        interpolated = multiply_dimensionwise(vandermonde, element_values)

        # Each element is placed on the uniform grid assuming reference directions align with
        # physical axes (ξ→x, η→y) and nodal values use the DGSEM tensor order along (ξ, η).
        # first_index[dim] refers to the first index of the nodes of this element within the array of global tensor nodes
        first_index = Vector{Int}(undef, 2)
        for dim in 1:2
            lower_left = normalized_coordinates[dim, element] -
                         (n_uniform_nodes - 1) / 2 * dx_global
            first_index[dim] = round(Int,
                                     (lower_left - (-1 + dx_global / 2)) /
                                     dx_global) + 1
        end

        # with `u_uniform` as the full uniform Cartesian grid over the domain, `r1` and `r2` are the
        # global grid indices corresponding to `u_uniform` that this specific element's interpolated block fits within
        # Essentially, `r1` and `r2` are the positions of the current element within `u_uniform`
        r1 = first_index[1]:(first_index[1] + n_uniform_nodes - 1)
        r2 = first_index[2]:(first_index[2] + n_uniform_nodes - 1)
        u_uniform[:, r1, r2] .= interpolated[:, :, :]
    end
    return u_uniform
end

"""
    compute_kinetic_energy_spectrum(u, mesh::DGMultiMesh{2}, equations,
                                    dg::DGMultiSBP, cache)

Compute the energy spectrum for a 2D `DGMulti` finite-difference SBP solution whose
nodes already form a uniform Cartesian grid.
"""
function compute_kinetic_energy_spectrum(u, mesh::DGMultiMesh{2},
                                         equations::AbstractCompressibleEulerEquations,
                                         dg::DGMultiSBP, cache)
    # Unpacks the primitive variables from the conservative state for FDSBP DGMulti solutions
    u_values = parent(u)
    n_points = length(u_values)
    n = round(Int, sqrt(n_points))
    q = cons2prim.(u_values, Ref(equations)) # q is the vector that contains the primitive variables for density and velocity converted from the conservative variables
    rho = reshape(getindex.(q, 1), n, n)
    density_weighted_velocity_1 = sqrt.(rho) .* reshape(getindex.(q, 2), n, n)
    density_weighted_velocity_2 = sqrt.(rho) .* reshape(getindex.(q, 3), n, n)

    return compute_kinetic_energy_spectrum(density_weighted_velocity_1,
                                           density_weighted_velocity_2)
end
end # @muladd
