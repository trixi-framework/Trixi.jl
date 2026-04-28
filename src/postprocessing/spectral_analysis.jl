# By default, Julia/LLVM does not use fused multiply-add operations (FMAs)
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details
@muladd begin
#! format: noindent

"""
    compute_energy_spectrum(sol; kwargs...)

Compute the energy spectrum from the final state of an ODE solution returned by
`solve`. Keyword arguments are forwarded to the semidiscretization-specific method.
Dispatch then selects the 2D or 3D implementation based on the mesh/solver types

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
The concrete method is dispatched to 2D or 3D based on the resulting argument types
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
Specialized implementations are provided in the 2D and 3D dispatch files
"""
function radial_energy_spectrum(energy_modes)
    # Convert the multi-dimensional Fourier energy into the
    # 1D isotropic spectrum E by summing all modes whose
    # wavenumber rounds to the same integer shell, accounting for the FFT convention
    ndims_energy_modes = ndims(energy_modes)
    maximum_wavenumber_squared = 0
    for dim in 1:ndims_energy_modes
        maximum_wavenumber_squared += (size(energy_modes, dim) ÷ 2)^2
    end
    maximum_wavenumber = round(Int, sqrt(maximum_wavenumber_squared))
    energy_spectrum = zeros(eltype(energy_modes), maximum_wavenumber + 1)
    wavenumbers = collect(0:maximum_wavenumber)

    # Iterate over all Fourier coefficients and bin each mode by isotropic shell
    for index in CartesianIndices(energy_modes)
        effective_wavenumber_squared = 0
        for dim in 1:ndims_energy_modes
            # Julia arrays are 1-based, while FFT mode numbers are centered around zero so subtract 1
            wavenumber = index[dim] - 1

            # Convert wrapped FFT ordering to signed integer wavenumbers
            # For an axis of length N, FFT bins correspond to
            # 0, 1, ..., floor(N/2), -floor((N-1)/2), ..., -1
            # so wraps around to handle this convention
            if wavenumber > size(energy_modes, dim) ÷ 2
                wavenumber -= size(energy_modes, dim)
            end

            # Accumulate ||k||^2 = kx^2 + ky^2 (+ kz^2 in 3D)
            effective_wavenumber_squared += wavenumber^2
        end

        # Radially bin modes by nearest integer shell index
        effective_wavenumber = sqrt(effective_wavenumber_squared)
        shell = round(Int, effective_wavenumber)
        if shell <= maximum_wavenumber
            energy_spectrum[shell + 1] += energy_modes[index]
        end
    end

    return energy_spectrum, wavenumbers
end

function dgmulti_primitive_variables(u, equations, dg, ::Val{NDIMS}) where {NDIMS}
    # Uses primitive variables from the solution vector
    u_values = StructArray(u)
    n_points = length(u_values)
    n_points_per_dimension = round(Int, n_points^(1 / NDIMS))
    if n_points_per_dimension^NDIMS != n_points
        throw(ArgumentError("DGMulti data does not form a uniform Cartesian grid"))
    end

    # Repack solution components into Cartesian arrays for the FFT kernel
    primitive_grid_size = ntuple(_ -> n_points_per_dimension, Val(NDIMS))
    primitive_variables = Vector{Array{real(dg), NDIMS}}(undef, NDIMS + 1)
    for variable in eachindex(primitive_variables)
        primitive_variables[variable] = Array{real(dg)}(undef, primitive_grid_size)
    end
    for index in eachindex(u_values)
        u_node = cons2prim(u_values[index], equations)
        for variable in eachindex(primitive_variables)
            primitive_variables[variable][index] = u_node[variable]
        end
    end

    return primitive_variables
end

include("spectral_analysis_2d.jl")
include("spectral_analysis_3d.jl")
end # @muladd
