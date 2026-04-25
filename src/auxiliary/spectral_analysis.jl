# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

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

Returns `(energy_spectrum, wavenumbers)`, where `energy_spectrum[i]` contains the energy in the
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
