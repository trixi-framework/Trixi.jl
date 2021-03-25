
# This file contains general numerical fluxes that are not specific to certain equations

"""
    flux_central(u_ll, u_rr, orientation, equations::AbstractEquations)

The classical central numerical flux `f((u_ll) + f(u_rr)) / 2`. When this flux is
used as volume flux, the discretization is equivalent to the classical weak form
DG method (except floating point errors).
"""
@inline function flux_central(u_ll, u_rr, orientation, equations::AbstractEquations)
  # Calculate regular 1D fluxes
  f_ll = flux(u_ll, orientation, equations)
  f_rr = flux(u_rr, orientation, equations)

  # Average regular fluxes
  return 0.5 * (f_ll + f_rr)
end


"""
    FluxPlusDissipation(numerical_flux, dissipation)

Combine a `numerical_flux` with a `dissipation` operator to create a new numerical flux.
"""
struct FluxPlusDissipation{NumericalFlux, Dissipation}
  numerical_flux::NumericalFlux
  dissipation::Dissipation
end

@inline function (numflux::FluxPlusDissipation{NumericalFlux, Dissipation})(u_ll, u_rr, orientation, equations) where {NumericalFlux, Dissipation}
  @unpack numerical_flux, dissipation = numflux

  return numerical_flux(u_ll, u_rr, orientation, equations) + dissipation(u_ll, u_rr, orientation, equations)
end


"""
    DissipationGlobalLaxFriedrichs(λ)

Create a global Lax-Friedrichs dissipation operator with dissipation coefficient `λ`.
"""
struct DissipationGlobalLaxFriedrichs{RealT}
  λ::RealT
end

@inline function (dissipation::DissipationGlobalLaxFriedrichs)(u_ll, u_rr, orientation, equations)
  @unpack λ = dissipation
  return -λ/2 * (u_rr - u_ll)
end


"""
    DissipationLocalLaxFriedrichs(max_abs_speed=max_abs_speed_naive)

Create a local Lax-Friedrichs dissipation operator where the maximum absolute wave speed
is estimated as `max_abs_speed(u_ll, u_rr, orientation, equations)`, defaulting to
[`max_abs_speed_naive`](@ref).
"""
struct DissipationLocalLaxFriedrichs{MaxAbsSpeed}
  max_abs_speed::MaxAbsSpeed
end

DissipationLocalLaxFriedrichs() = DissipationLocalLaxFriedrichs(max_abs_speed_naive)

@inline function (dissipation::DissipationLocalLaxFriedrichs{MaxAbsSpeed})(u_ll, u_rr, orientation, equations) where {MaxAbsSpeed}
  λ = dissipation.max_abs_speed(u_ll, u_rr, orientation, equations)
  return -0.5 * λ * (u_rr - u_ll)
end


"""
    max_abs_speed_naive(u_ll, u_rr, orientation, equations)

Simple and fast estimate of the maximal wave speed of the Riemann problem with left and right states
`u_ll, u_rr`, based only on the local wave speeds associated to `u_ll` and `u_rr`.
"""
function max_abs_speed_naive end


"""
    FluxLaxFriedrichs(max_abs_speed=max_abs_speed_naive)

Local Lax-Friedrichs (Rusanov) flux with maximum wave speed estimate provided by
`max_abs_speed`, cf. [`DissipationLocalLaxFriedrichs`](@ref) and
[`max_abs_speed_naive`](@ref).
"""
function FluxLaxFriedrichs(max_abs_speed=max_abs_speed_naive)
  FluxPlusDissipation(flux_central, DissipationLocalLaxFriedrichs(max_abs_speed))
end
