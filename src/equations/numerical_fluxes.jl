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
    DissipationLocalLaxFriedrichs(max_abs_speed=max_abs_speed)

Create a local Lax-Friedrichs dissipation operator where the maximum absolute wave speed
is estimated as
`max_abs_speed(u_ll, u_rr, orientation_or_normal_direction, equations)`,
defaulting to [`max_abs_speed`](@ref).
"""
struct DissipationLocalLaxFriedrichs{MaxAbsSpeed}
    max_abs_speed::MaxAbsSpeed
end

DissipationLocalLaxFriedrichs() = DissipationLocalLaxFriedrichs(max_abs_speed)

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
    max_abs_speed_naive(u_ll, u_rr, orientation::Integer, equations)
    max_abs_speed_naive(u_ll, u_rr, normal_direction::AbstractVector, equations)

Simple and fast estimate of the maximal wave speed of the Riemann problem with left and right states
`u_ll, u_rr`, based only on the local wave speeds associated to `u_ll` and `u_rr`.

For non-integer arguments `normal_direction` in one dimension, `max_abs_speed_naive` returns
`abs(normal_direction[1]) * max_abs_speed_naive(u_ll, u_rr, 1, equations)`.

Slightly more diffusive/overestimating than [`max_abs_speed`](@ref).
"""
function max_abs_speed_naive end

# for non-integer `orientation_or_normal` arguments.
@inline function max_abs_speed_naive(u_ll, u_rr, normal_direction::AbstractVector,
                                     equations::AbstractEquations{1})
    return abs(normal_direction[1]) * max_abs_speed_naive(u_ll, u_rr, 1, equations)
end

"""
    max_abs_speed(u_ll, u_rr, orientation::Integer, equations)
    max_abs_speed(u_ll, u_rr, normal_direction::AbstractVector, equations)

Simple and fast estimate of the maximal wave speed of the Riemann problem with left and right states
`u_ll, u_rr`, based only on the local wave speeds associated to `u_ll` and `u_rr`.
Less diffusive, i.e., overestimating than [`max_abs_speed_naive`](@ref).

In particular, `max_abs_speed(u, u, i, equations)` gives the same result as `max_abs_speeds(u, equations)[i]`,
i.e., the wave speeds used in `max_dt` which computes the maximum stable time step size through the 
[`StepsizeCallback`](@ref).

For non-integer arguments `normal_direction` in one dimension, `max_abs_speed_naive` returns
`abs(normal_direction[1]) * max_abs_speed_naive(u_ll, u_rr, 1, equations)`.

Defaults to [`min_max_speed_naive`](@ref) if no specialized version for the 'equations` at hand is available.
"""
@inline function max_abs_speed(u_ll, u_rr,
                               orientation_or_normal_direction,
                               equations::AbstractEquations)
    # Use naive version as "backup" if no specialized version for the equations at hand is available                                             
    max_abs_speed_naive(u_ll, u_rr, orientation_or_normal_direction, equations)
end

const FluxLaxFriedrichs{MaxAbsSpeed} = FluxPlusDissipation{typeof(flux_central),
                                                           DissipationLocalLaxFriedrichs{MaxAbsSpeed}}
"""
    FluxLaxFriedrichs(max_abs_speed=max_abs_speed)

Local Lax-Friedrichs (Rusanov) flux with maximum wave speed estimate provided by
`max_abs_speed`, cf. [`DissipationLocalLaxFriedrichs`](@ref) and
[`max_abs_speed_naive`](@ref).
"""
function FluxLaxFriedrichs(max_abs_speed = max_abs_speed)
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

@doc raw"""
    DissipationLaxFriedrichsEntropyVariables(max_abs_speed=max_abs_speed)

Create a local Lax-Friedrichs-type dissipation operator that is provably entropy stable. This operator
must be used together with an entropy-conservative two-point flux function (e.g., `flux_ec`) to yield 
an entropy-stable surface flux. The surface flux function can be initialized as:
```julia
flux_es = FluxPlusDissipation(flux_ec, DissipationLaxFriedrichsEntropyVariables())
```

In particular, the numerical flux has the form
```math
f^{\mathrm{ES}} = f^{\mathrm{EC}} - \frac{1}{2} \lambda_{\mathrm{max}} H (w_r - w_l),
```
where ``f^{\mathrm{EC}}`` is the entropy-conservative two-point flux function (computed with, e.g., `flux_ec`), ``\lambda_{\mathrm{max}}`` 
is the maximum wave speed estimated as `max_abs_speed(u_l, u_r, orientation_or_normal_direction, equations)`,
defaulting to [`max_abs_speed`](@ref), ``H`` is a symmetric positive-definite dissipation matrix that
depends on the left and right states `u_l` and `u_r`, and ``(w_r - w_l)`` is the jump in entropy variables.
Ideally, ``H (w_r - w_l) = (u_r - u_l)``, such that the dissipation operator is consistent with the local
Lax-Friedrichs dissipation.

The entropy-stable dissipation operator is computed with the function
`function (dissipation::DissipationLaxFriedrichsEntropyVariables)(u_l, u_r, orientation_or_normal_direction, equations)`,
which must be specialized for each equation.

For the derivation of the dissipation matrix for the multi-ion GLM-MHD equations, see:
- A. Rueda-Ramírez, A. Sikstel, G. Gassner, An Entropy-Stable Discontinuous Galerkin Discretization
  of the Ideal Multi-Ion Magnetohydrodynamics System (2024). Journal of Computational Physics.
  [DOI: 10.1016/j.jcp.2024.113655](https://doi.org/10.1016/j.jcp.2024.113655).
"""
struct DissipationLaxFriedrichsEntropyVariables{MaxAbsSpeed}
    max_abs_speed::MaxAbsSpeed
end

DissipationLaxFriedrichsEntropyVariables() = DissipationLaxFriedrichsEntropyVariables(max_abs_speed)

function Base.show(io::IO, d::DissipationLaxFriedrichsEntropyVariables)
    print(io, "DissipationLaxFriedrichsEntropyVariables(", d.max_abs_speed, ")")
end

@doc raw"""
    DissipationMatrixWintersEtal()

Creates the Roe-like entropy-stable matrix dissipation operator from Winters et al. (2017). This operator
must be used together with an entropy-conservative two-point flux function 
(e.g., [`flux_ranocha`](@ref) or [`flux_chandrashekar`](@ref)) to yield 
an entropy-stable surface flux. The surface flux function can be initialized as:
```julia
flux_es = FluxPlusDissipation(flux_ec, DissipationMatrixWintersEtal())
```
The 1D and 2D implementations are adapted from the [Atum.jl library](https://github.com/mwarusz/Atum.jl/blob/c7ed44f2b7972ac726ef345da7b98b0bda60e2a3/src/balancelaws/euler.jl#L198).
The 3D implementation is adapted from the [FLUXO library](https://github.com/project-fluxo/fluxo)

For the derivation of the matrix dissipation operator, see:
- A. R. Winters, D. Derigs, G. Gassner, S. Walch, A uniquely defined entropy stable matrix dissipation operator 
  for high Mach number ideal MHD and compressible Euler simulations (2017). Journal of Computational Physics.
  [DOI: 10.1016/j.jcp.2016.12.006](https://doi.org/10.1016/j.jcp.2016.12.006).
"""
struct DissipationMatrixWintersEtal end

@inline function (dissipation::DissipationMatrixWintersEtal)(u_ll, u_rr,
                                                             orientation::Integer,
                                                             equations::AbstractEquations{1})
    return dissipation(u_ll, u_rr, SVector(1), equations)
end

@inline function (dissipation::DissipationMatrixWintersEtal)(u_ll, u_rr,
                                                             orientation::Integer,
                                                             equations::AbstractEquations{2})
    if orientation == 1
        return dissipation(u_ll, u_rr, SVector(1, 0), equations)
    else # orientation == 2
        return dissipation(u_ll, u_rr, SVector(0, 1), equations)
    end
end

@inline function (dissipation::DissipationMatrixWintersEtal)(u_ll, u_rr,
                                                             orientation::Integer,
                                                             equations::AbstractEquations{3})
    if orientation == 1
        return dissipation(u_ll, u_rr, SVector(1, 0, 0), equations)
    elseif orientation == 2
        return dissipation(u_ll, u_rr, SVector(0, 1, 0), equations)
    else # orientation == 3
        return dissipation(u_ll, u_rr, SVector(0, 0, 1), equations)
    end
end

"""
    DissipationEntropyStable(max_abs_speed=max_abs_speed_naive)

Create a local Lax-Friedrichs-type dissipation operator that is provably entropy stable. This operator
must be used together with an entropy-conservative two-point flux function (e.g., `flux_ec`) to yield 
an entropy-stable surface flux. The surface flux function can be initialized as:
```
flux_es = FluxPlusDissipation(flux_ec, DissipationEntropyStable())
```

In particular, the numerical flux has the form
```math
f^{ES} = f^{EC} + \frac{1}{2} λ_{max} H (w_r - w_l),
````
where ``f^{EC}`` is the entropy-conservative two-point flux function (computed with, e.g., `flux_ec`), ``λ_{max}`` 
is the maximum wave speed estimated as `max_abs_speed(u_l, u_r, orientation_or_normal_direction, equations)`,
defaulting to [`max_abs_speed_naive`](@ref), ``H`` is a symmetric positive-definite dissipation matrix that
depends on the left and right states `u_l` and `u_r`, and ``(w_r - w_l)`` is the jump in entropy variables.
Ideally, ``H (w_r - w_l) = (u_r - u_l)``, such that the dissipation operator is consistent with the local
Lax-Friedrichs dissipation.

The entropy-stable dissipation operator is computed with the function
`function (dissipation::DissipationEntropyStable)(u_l, u_r, orientation_or_normal_direction, equations)`,
which must be specialized for each equation.

For the derivation of the dissipation matrix for the multi-ion GLM-MHD equations, see:
- A. Rueda-Ramírez, A. Sikstel, G. Gassner, An Entropy-Stable Discontinuous Galerkin Discretization
  of the Ideal Multi-Ion Magnetohydrodynamics System (2024). Journal of Computational Physics.
  [DOI: 10.1016/j.jcp.2024.113655](https://doi.org/10.1016/j.jcp.2024.113655).
"""
struct DissipationEntropyStable{MaxAbsSpeed}
    max_abs_speed::MaxAbsSpeed
end

DissipationEntropyStable() = DissipationEntropyStable(max_abs_speed_naive)

function Base.show(io::IO, d::DissipationEntropyStable)
    print(io, "DissipationEntropyStable(", d.max_abs_speed, ")")
end

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
    flux_wintermeyer_etal(u_ll, u_rr, orientation_or_normal_direction, equations)

Total energy conservative (mathematical entropy for shallow water equations) split form.
When the bottom topography is nonzero this scheme will be well-balanced when used as a `volume_flux`.
For the `surface_flux` either [`flux_wintermeyer_etal`](@ref) or [`flux_fjordholm_etal`](@ref) can
be used to ensure well-balancedness and entropy conservation.

!!! note
    This function is defined in Trixi.jl to have a common interface for the
    methods implemented in the subpackages [TrixiAtmo.jl](https://github.com/trixi-framework/TrixiAtmo.jl) 
    and [TrixiShallowWater.jl](https://github.com/trixi-framework/TrixiShallowWater.jl).
"""
function flux_wintermeyer_etal end

"""
    flux_nonconservative_wintermeyer_etal(u_ll, u_rr, orientation_or_normal_direction, equations)

Non-symmetric two-point volume flux discretizing the nonconservative (source) term
that contains the gradient of the bottom topography for the shallow water equations.

Gives entropy conservation and well-balancedness on both the volume and surface when combined with
[`flux_wintermeyer_etal`](@ref).

!!! note
    This function is defined in Trixi.jl to have a common interface for the
    methods implemented in the subpackages [TrixiAtmo.jl](https://github.com/trixi-framework/TrixiAtmo.jl) 
    and [TrixiShallowWater.jl](https://github.com/trixi-framework/TrixiShallowWater.jl).
"""
function flux_nonconservative_wintermeyer_etal end

"""
    flux_fjordholm_etal(u_ll, u_rr, orientation_or_normal_direction, equations)

Total energy conservative (mathematical entropy for shallow water equations). When the bottom topography
is nonzero this should only be used as a surface flux otherwise the scheme will not be well-balanced.
For well-balancedness in the volume flux use [`flux_wintermeyer_etal`](@ref).

!!! note
    This function is defined in Trixi.jl to have a common interface for the
    methods implemented in the subpackages [TrixiAtmo.jl](https://github.com/trixi-framework/TrixiAtmo.jl) 
    and [TrixiShallowWater.jl](https://github.com/trixi-framework/TrixiShallowWater.jl).
"""
function flux_fjordholm_etal end

"""
    flux_nonconservative_fjordholm_etal(u_ll, u_rr, orientation_or_normal_direction, equations)

Non-symmetric two-point surface flux discretizing the nonconservative (source) term of
that contains the gradient of the bottom topography for the shallow water equations.

This flux can be used together with [`flux_fjordholm_etal`](@ref) at interfaces to ensure entropy
conservation and well-balancedness.

!!! note
    This function is defined in Trixi.jl to have a common interface for the
    methods implemented in the subpackages [TrixiAtmo.jl](https://github.com/trixi-framework/TrixiAtmo.jl) 
    and [TrixiShallowWater.jl](https://github.com/trixi-framework/TrixiShallowWater.jl).
"""
function flux_nonconservative_fjordholm_etal end

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
