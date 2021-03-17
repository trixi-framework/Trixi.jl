
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


# TODO: Must be documented and might need a better name
struct FluxComparedToCentral{Numflux}
  numflux::Numflux
end

@inline function (f::FluxComparedToCentral{Numflux})(u_ll, u_rr, orientation, equations) where {Numflux}

  f_baseline = f.numflux(u_ll, u_rr, orientation, equations)
  f_central = flux_central(u_ll, u_rr, orientation, equations)
  w_ll = cons2entropy(u_ll, equations)
  w_rr = cons2entropy(u_rr, equations)
  # The local entropy production of a numerical flux at an interface is
  #   dot(w_rr - w_ll, numerical_flux) - (psi_rr - psi_ll),
  # see Tadmor (1987). Since the flux potential is the same for both, we can
  # omit that part.
  delta_entropy = dot(w_rr - w_ll, f_central - f_baseline)
  if delta_entropy < 0
    return f_central
  else
    return 2 * f_baseline - f_central
  end
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
