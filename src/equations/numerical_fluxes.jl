# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

# This file contains general numerical fluxes that are not specific to certain equations

"""
    flux_central(u_ll, u_rr, orientation_or_normal_direction, equations::AbstractEquations)

The classical central numerical flux `f((u_ll) + f(u_rr)) / 2`. When this flux is
used as volume flux, the discretization is equivalent to the classical weak form
DG method (except floating point errors).
"""
@inline function flux_central(u_ll, u_rr, orientation_or_normal_direction,
                              equations::AbstractEquations)
    # Calculate regular 1D fluxes
    f_ll = flux(u_ll, orientation_or_normal_direction, equations)
    f_rr = flux(u_rr, orientation_or_normal_direction, equations)

    # Average regular fluxes
    return 0.5f0 * (f_ll + f_rr)
end

"""
    FluxPlusDissipation(numerical_flux, dissipation)

Combine a `numerical_flux` with a `dissipation` operator to create a new numerical flux.
"""
struct FluxPlusDissipation{NumericalFlux, Dissipation}
    numerical_flux::NumericalFlux
    dissipation::Dissipation
end

@inline function (numflux::FluxPlusDissipation)(u_ll, u_rr,
                                                orientation_or_normal_direction,
                                                equations)
    @unpack numerical_flux, dissipation = numflux

    return (numerical_flux(u_ll, u_rr, orientation_or_normal_direction, equations)
            +
            dissipation(u_ll, u_rr, orientation_or_normal_direction, equations))
end

function Base.show(io::IO, f::FluxPlusDissipation)
    print(io, "FluxPlusDissipation(", f.numerical_flux, ", ", f.dissipation, ")")
end

"""
    FluxRotated(numerical_flux)

Compute a `numerical_flux` flux in direction of a normal vector by rotating the solution,
computing the numerical flux in x-direction, and rotating the calculated flux back.

Requires a rotationally invariant equation with equation-specific functions
[`rotate_to_x`](@ref) and [`rotate_from_x`](@ref).
"""
struct FluxRotated{NumericalFlux}
    numerical_flux::NumericalFlux
end

# Rotated surface flux computation (2D version)
@inline function (flux_rotated::FluxRotated)(u,
                                             normal_direction::AbstractVector,
                                             equations::AbstractEquations{2})
    @unpack numerical_flux = flux_rotated

    norm_ = norm(normal_direction)
    # Normalize the vector without using `normalize` since we need to multiply by the `norm_` later
    normal_vector = normal_direction / norm_

    u_rotated = rotate_to_x(u, normal_vector, equations)

    f = numerical_flux(u_rotated, 1, equations)

    return rotate_from_x(f, normal_vector, equations) * norm_
end

# Rotated surface flux computation (2D version)
@inline function (flux_rotated::FluxRotated)(u_ll, u_rr,
                                             normal_direction::AbstractVector,
                                             equations::AbstractEquations{2})
    @unpack numerical_flux = flux_rotated

    norm_ = norm(normal_direction)
    # Normalize the vector without using `normalize` since we need to multiply by the `norm_` later
    normal_vector = normal_direction / norm_

    u_ll_rotated = rotate_to_x(u_ll, normal_vector, equations)
    u_rr_rotated = rotate_to_x(u_rr, normal_vector, equations)

    f = numerical_flux(u_ll_rotated, u_rr_rotated, 1, equations)

    return rotate_from_x(f, normal_vector, equations) * norm_
end

# Rotated surface flux computation (3D version)
@inline function (flux_rotated::FluxRotated)(u_ll, u_rr,
                                             normal_direction::AbstractVector,
                                             equations::AbstractEquations{3})
    @unpack numerical_flux = flux_rotated

    # Storing these vectors could increase the performance by 20 percent
    norm_ = norm(normal_direction)
    # Normalize the vector without using `normalize` since we need to multiply by the `norm_` later
    normal_vector = normal_direction / norm_

    # Some vector that can't be identical to normal_vector (unless normal_vector == 0)
    tangent1 = SVector(normal_direction[2], normal_direction[3], -normal_direction[1])
    # Orthogonal projection
    tangent1 -= dot(normal_vector, tangent1) * normal_vector
    tangent1 = normalize(tangent1)

    # Third orthogonal vector
    tangent2 = normalize(cross(normal_direction, tangent1))

    u_ll_rotated = rotate_to_x(u_ll, normal_vector, tangent1, tangent2, equations)
    u_rr_rotated = rotate_to_x(u_rr, normal_vector, tangent1, tangent2, equations)

    f = numerical_flux(u_ll_rotated, u_rr_rotated, 1, equations)

    return rotate_from_x(f, normal_vector, tangent1, tangent2, equations) * norm_
end

Base.show(io::IO, f::FluxRotated) = print(io, "FluxRotated(", f.numerical_flux, ")")

"""
    DissipationGlobalLaxFriedrichs(λ)

Create a global Lax-Friedrichs dissipation operator with dissipation coefficient `λ`.
"""
struct DissipationGlobalLaxFriedrichs{RealT}
    λ::RealT
end

@inline function (dissipation::DissipationGlobalLaxFriedrichs)(u_ll, u_rr,
                                                               orientation::Integer,
                                                               equations)
    @unpack λ = dissipation
    return -λ / 2 * (u_rr - u_ll)
end

@inline function (dissipation::DissipationGlobalLaxFriedrichs)(u_ll, u_rr,
                                                               normal_direction::AbstractVector,
                                                               equations)
    @unpack λ = dissipation
    return -λ / 2 * norm(normal_direction) * (u_rr - u_ll)
end

function Base.show(io::IO, d::DissipationGlobalLaxFriedrichs)
    print(io, "DissipationGlobalLaxFriedrichs(", d.λ, ")")
end

"""
    DissipationLocalLaxFriedrichs(max_abs_speed=max_abs_speed_naive)

Create a local Lax-Friedrichs dissipation operator where the maximum absolute wave speed
is estimated as
`max_abs_speed(u_ll, u_rr, orientation_or_normal_direction, equations)`,
defaulting to [`max_abs_speed_naive`](@ref).
"""
struct DissipationLocalLaxFriedrichs{MaxAbsSpeed}
    max_abs_speed::MaxAbsSpeed
end

DissipationLocalLaxFriedrichs() = DissipationLocalLaxFriedrichs(max_abs_speed_naive)

@inline function (dissipation::DissipationLocalLaxFriedrichs)(u_ll, u_rr,
                                                              orientation_or_normal_direction,
                                                              equations)
    λ = dissipation.max_abs_speed(u_ll, u_rr, orientation_or_normal_direction,
                                  equations)
    return -0.5f0 * λ * (u_rr - u_ll)
end

function Base.show(io::IO, d::DissipationLocalLaxFriedrichs)
    print(io, "DissipationLocalLaxFriedrichs(", d.max_abs_speed, ")")
end

"""
    max_abs_speed_naive(u_ll, u_rr, orientation::Integer,   equations)
    max_abs_speed_naive(u_ll, u_rr, normal_direction::AbstractVector, equations)

Simple and fast estimate of the maximal wave speed of the Riemann problem with left and right states
`u_ll, u_rr`, based only on the local wave speeds associated to `u_ll` and `u_rr`.

For non-integer arguments `normal_direction` in one dimension, `max_abs_speed_naive` returns
`abs(normal_direction[1]) * max_abs_speed_naive(u_ll, u_rr, 1, equations)`.
"""
function max_abs_speed_naive end

# for non-integer `orientation_or_normal` arguments.
@inline function max_abs_speed_naive(u_ll, u_rr, normal_direction::AbstractVector,
                                     equations::AbstractEquations{1})
    return abs(normal_direction[1]) * max_abs_speed_naive(u_ll, u_rr, 1, equations)
end

const FluxLaxFriedrichs{MaxAbsSpeed} = FluxPlusDissipation{typeof(flux_central),
                                                           DissipationLocalLaxFriedrichs{MaxAbsSpeed}}
"""
    FluxLaxFriedrichs(max_abs_speed=max_abs_speed_naive)

Local Lax-Friedrichs (Rusanov) flux with maximum wave speed estimate provided by
`max_abs_speed`, cf. [`DissipationLocalLaxFriedrichs`](@ref) and
[`max_abs_speed_naive`](@ref).
"""
function FluxLaxFriedrichs(max_abs_speed = max_abs_speed_naive)
    FluxPlusDissipation(flux_central, DissipationLocalLaxFriedrichs(max_abs_speed))
end

function Base.show(io::IO, f::FluxLaxFriedrichs)
    print(io, "FluxLaxFriedrichs(", f.dissipation.max_abs_speed, ")")
end

"""
    flux_lax_friedrichs

See [`FluxLaxFriedrichs`](@ref).
"""
const flux_lax_friedrichs = FluxLaxFriedrichs()

"""
    FluxHLL(min_max_speed=min_max_speed_davis)

Create an HLL (Harten, Lax, van Leer) numerical flux where the minimum and maximum
wave speeds are estimated as
`λ_min, λ_max = min_max_speed(u_ll, u_rr, orientation_or_normal_direction, equations)`,
defaulting to [`min_max_speed_davis`](@ref).
Original paper:
- Amiram Harten, Peter D. Lax, Bram van Leer (1983)
  On Upstream Differencing and Godunov-Type Schemes for Hyperbolic Conservation Laws
  [DOI: 10.1137/1025002](https://doi.org/10.1137/1025002)
"""
struct FluxHLL{MinMaxSpeed}
    min_max_speed::MinMaxSpeed
end

FluxHLL() = FluxHLL(min_max_speed_davis)

"""
    min_max_speed_naive(u_ll, u_rr, orientation::Integer, equations)
    min_max_speed_naive(u_ll, u_rr, normal_direction::AbstractVector, equations)

Simple and fast estimate(!) of the minimal and maximal wave speed of the Riemann problem with
left and right states `u_ll, u_rr`, usually based only on the local wave speeds associated to
`u_ll` and `u_rr`.
Slightly more diffusive than [`min_max_speed_davis`](@ref).
- Amiram Harten, Peter D. Lax, Bram van Leer (1983)
  On Upstream Differencing and Godunov-Type Schemes for Hyperbolic Conservation Laws
  [DOI: 10.1137/1025002](https://doi.org/10.1137/1025002)

See eq. (10.37) from
- Eleuterio F. Toro (2009)
  Riemann Solvers and Numerical Methods for Fluid Dynamics: A Practical Introduction
  [DOI: 10.1007/b79761](https://doi.org/10.1007/b79761)

See also [`FluxHLL`](@ref), [`min_max_speed_davis`](@ref), [`min_max_speed_einfeldt`](@ref).
"""
function min_max_speed_naive end

"""
    min_max_speed_davis(u_ll, u_rr, orientation::Integer, equations)
    min_max_speed_davis(u_ll, u_rr, normal_direction::AbstractVector, equations)

Simple and fast estimates of the minimal and maximal wave speed of the Riemann problem with
left and right states `u_ll, u_rr`, usually based only on the local wave speeds associated to
`u_ll` and `u_rr`.

- S.F. Davis (1988)
  Simplified Second-Order Godunov-Type Methods
  [DOI: 10.1137/0909030](https://doi.org/10.1137/0909030)

See eq. (10.38) from
- Eleuterio F. Toro (2009)
  Riemann Solvers and Numerical Methods for Fluid Dynamics: A Practical Introduction
  [DOI: 10.1007/b79761](https://doi.org/10.1007/b79761)
See also [`FluxHLL`](@ref), [`min_max_speed_naive`](@ref), [`min_max_speed_einfeldt`](@ref).
"""
function min_max_speed_davis end

"""
    min_max_speed_einfeldt(u_ll, u_rr, orientation::Integer, equations)
    min_max_speed_einfeldt(u_ll, u_rr, normal_direction::AbstractVector, equations)

More advanced mininmal and maximal wave speed computation based on
- Bernd Einfeldt (1988)
  On Godunov-type methods for gas dynamics.
  [DOI: 10.1137/0725021](https://doi.org/10.1137/0725021)
- Bernd Einfeldt, Claus-Dieter Munz, Philip L. Roe and Björn Sjögreen (1991)
  On Godunov-type methods near low densities.
  [DOI: 10.1016/0021-9991(91)90211-3](https://doi.org/10.1016/0021-9991(91)90211-3)

originally developed for the compressible Euler equations.
A compact representation can be found in [this lecture notes, eq. (9.28)](https://metaphor.ethz.ch/x/2019/hs/401-4671-00L/literature/mishra_hyperbolic_pdes.pdf).

See also [`FluxHLL`](@ref), [`min_max_speed_naive`](@ref), [`min_max_speed_davis`](@ref).
"""
function min_max_speed_einfeldt end

@inline function (numflux::FluxHLL)(u_ll, u_rr, orientation_or_normal_direction,
                                    equations)
    λ_min, λ_max = numflux.min_max_speed(u_ll, u_rr, orientation_or_normal_direction,
                                         equations)

    if λ_min >= 0 && λ_max >= 0
        return flux(u_ll, orientation_or_normal_direction, equations)
    elseif λ_max <= 0 && λ_min <= 0
        return flux(u_rr, orientation_or_normal_direction, equations)
    else
        f_ll = flux(u_ll, orientation_or_normal_direction, equations)
        f_rr = flux(u_rr, orientation_or_normal_direction, equations)
        inv_λ_max_minus_λ_min = inv(λ_max - λ_min)
        factor_ll = λ_max * inv_λ_max_minus_λ_min
        factor_rr = λ_min * inv_λ_max_minus_λ_min
        factor_diss = λ_min * λ_max * inv_λ_max_minus_λ_min
        return factor_ll * f_ll - factor_rr * f_rr + factor_diss * (u_rr - u_ll)
    end
end

Base.show(io::IO, numflux::FluxHLL) = print(io, "FluxHLL(", numflux.min_max_speed, ")")

"""
    flux_hll

See [`FluxHLL`](@ref).
"""
const flux_hll = FluxHLL()

"""
    flux_hlle

See [`min_max_speed_einfeldt`](@ref).
This is a [`FluxHLL`](@ref)-type two-wave solver with special estimates of the wave speeds.
"""
const flux_hlle = FluxHLL(min_max_speed_einfeldt)

"""
    flux_shima_etal_turbo(u_ll, u_rr, orientation_or_normal_direction, equations)

Equivalent to [`flux_shima_etal`](@ref) except that it may use specialized
methods, e.g., when used with [`VolumeIntegralFluxDifferencing`](@ref).
These specialized methods may enable better use of SIMD instructions to
increase runtime efficiency on modern hardware.
"""
@inline function flux_shima_etal_turbo(u_ll, u_rr, orientation_or_normal_direction,
                                       equations)
    flux_shima_etal(u_ll, u_rr, orientation_or_normal_direction, equations)
end

"""
    flux_ranocha_turbo(u_ll, u_rr, orientation_or_normal_direction, equations)

Equivalent to [`flux_ranocha`](@ref) except that it may use specialized
methods, e.g., when used with [`VolumeIntegralFluxDifferencing`](@ref).
These specialized methods may enable better use of SIMD instructions to
increase runtime efficiency on modern hardware.
"""
@inline function flux_ranocha_turbo(u_ll, u_rr, orientation_or_normal_direction,
                                    equations)
    flux_ranocha(u_ll, u_rr, orientation_or_normal_direction, equations)
end

"""
    FluxHydrostaticReconstruction(numerical_flux, hydrostatic_reconstruction)

!!! warning "Experimental code"
    This numerical flux is experimental and may change in any future release.

Allow for some kind of hydrostatic reconstruction of the solution state prior to the
surface flux computation. This is a particular strategy to ensure that the method remains
well-balanced for the shallow water equations, see [`ShallowWaterEquations1D`](@ref)
or [`ShallowWaterEquations2D`](@ref).

For example, the hydrostatic reconstruction from Audusse et al. is implemented
in one and two spatial dimensions, see [`hydrostatic_reconstruction_audusse_etal`](@ref) or
the original paper
- Emmanuel Audusse, François Bouchut, Marie-Odile Bristeau, Rupert Klein, and Benoit Perthame (2004)
  A fast and stable well-balanced scheme with hydrostatic reconstruction for shallow water flows
  [DOI: 10.1137/S1064827503431090](https://doi.org/10.1137/S1064827503431090)

Other hydrostatic reconstruction techniques are available, particularly to handle wet / dry
fronts. A good overview of the development and application of hydrostatic reconstruction can be found in
- Guoxian Chen and Sebastian Noelle
  A unified surface-gradient and hydrostatic reconstruction scheme for the shallow water equations (2021)
  [RWTH Aachen preprint](https://www.igpm.rwth-aachen.de/forschung/preprints/517)
- Andreas Buttinger-Kreuzhuber, Zsolt Horváth, Sebastian Noelle, Günter Blöschl and Jürgen Waser (2019)
  A fast second-order shallow water scheme on two-dimensional structured grids over abrupt topography
  [DOI: 10.1016/j.advwatres.2019.03.010](https://doi.org/10.1016/j.advwatres.2019.03.010)
"""
struct FluxHydrostaticReconstruction{NumericalFlux, HydrostaticReconstruction}
    numerical_flux::NumericalFlux
    hydrostatic_reconstruction::HydrostaticReconstruction
end

@inline function (numflux::FluxHydrostaticReconstruction)(u_ll, u_rr,
                                                          orientation_or_normal_direction,
                                                          equations::AbstractEquations)
    @unpack numerical_flux, hydrostatic_reconstruction = numflux

    # Create the reconstructed left/right solution states in conservative form
    u_ll_star, u_rr_star = hydrostatic_reconstruction(u_ll, u_rr, equations)

    # Use the reconstructed states to compute the numerical surface flux
    return numerical_flux(u_ll_star, u_rr_star, orientation_or_normal_direction,
                          equations)
end

"""
    FluxUpwind(splitting)

A numerical flux `f(u_left, u_right) = f⁺(u_left) + f⁻(u_right)` based on
flux vector splitting.

The [`SurfaceIntegralUpwind`](@ref) with a given `splitting` is equivalent to
the [`SurfaceIntegralStrongForm`](@ref) with `FluxUpwind(splitting)`
as numerical flux (up to floating point differences). Note, that
[`SurfaceIntegralUpwind`](@ref) is only available on [`TreeMesh`](@ref).

!!! warning "Experimental implementation (upwind SBP)"
    This is an experimental feature and may change in future releases.
"""
struct FluxUpwind{Splitting}
    splitting::Splitting
end

@inline function (numflux::FluxUpwind)(u_ll, u_rr, orientation::Int, equations)
    @unpack splitting = numflux
    fm = splitting(u_rr, Val{:minus}(), orientation, equations)
    fp = splitting(u_ll, Val{:plus}(), orientation, equations)
    return fm + fp
end

@inline function (numflux::FluxUpwind)(u_ll, u_rr,
                                       normal_direction::AbstractVector,
                                       equations::AbstractEquations{2})
    @unpack splitting = numflux
    f_tilde_m = splitting(u_rr, Val{:minus}(), normal_direction, equations)
    f_tilde_p = splitting(u_ll, Val{:plus}(), normal_direction, equations)
    return f_tilde_m + f_tilde_p
end

Base.show(io::IO, f::FluxUpwind) = print(io, "FluxUpwind(", f.splitting, ")")
end # @muladd
