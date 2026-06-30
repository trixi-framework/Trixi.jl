# By default, Julia/LLVM does not use fused multiply-add operations (FMAs)
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details
@muladd begin
#! format: noindent

"""
    compute_kinetic_energy_spectrum(sol)

Compute the isotropic kinetic energy spectrum from the final state of an ODE
solution returned by `solve`.
"""
function compute_kinetic_energy_spectrum(sol)
    return compute_kinetic_energy_spectrum(sol.u[end], sol.prob.p)
end

"""
    compute_kinetic_energy_spectrum(u_ode, semi)

Compute the isotropic kinetic energy spectrum from an ODE state vector `u_ode`
and a Trixi.jl semidiscretization `semi`.

Currently, implemented methods are restricted to
`AbstractCompressibleEulerEquations` for
- `TreeMesh` + `DGSEM` data (via interpolation from LGL nodes to a uniform
  Cartesian grid), and
- `DGMultiMesh` + `DGMultiSBP` data (already sampled on a Cartesian grid)

## Returns
- `wavenumbers`: vector of matching 0-based integer wavenumber shell labels
- `energy_spectrum`: 1D vector holding the isotropic kinetic energy spectrum
  `E(k)` binned by integer wavenumber shell

## Constructs internally
- For DGSEM `TreeMesh` data, it interpolates **conservative** variables from LGL nodes to a
  uniform Cartesian grid, applies `cons2prim` at each uniform node, then applies FFTs
- For DGMulti `DGMultiMesh` data, nodal conservative values are converted with `cons2prim`
  on the existing Cartesian grid before FFTs
- It forms density-weighted velocity fields `sqrt(rho) * v_i`, computes
  Fourier-space kinetic energy from the FFT results, normalizes modal energy by
  the squared number of grid points, and radially bins wrapped
  FFT modes to form the final 1D isotropic spectrum `E(k)`

## References

- Winters, Moura, Mengaldo, Gassner, Walch, Peiro, et al. (2018)
  A comparative study on polynomial dealiasing and split form discontinuous
  Galerkin schemes for under-resolved turbulence computations
  [DOI: 10.1016/j.jcp.2018.06.016](https://doi.org/10.1016/j.jcp.2018.06.016)
"""
function compute_kinetic_energy_spectrum(u_ode,
                                         semi::AbstractSemidiscretization)
    return compute_kinetic_energy_spectrum(wrap_array_native(u_ode, semi),
                                           mesh_equations_solver_cache(semi)...)
end

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
    wavenumbers = collect(0:maximum_wavenumber) # Add zero wave number

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

    return wavenumbers, energy_spectrum
end

include("spectral_analysis_2d.jl")
include("spectral_analysis_3d.jl")
end # @muladd
