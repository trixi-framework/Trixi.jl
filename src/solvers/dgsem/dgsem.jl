# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

# Include utilities
include("interpolation.jl")
include("l2projection.jl")
include("basis_lobatto_legendre.jl")
include("basis_gauss_legendre.jl")

"""
    DGSEM(; RealT=Float64,
            polydeg::Integer,
            basis_type = LobattoLegendreBasis,
            surface_flux=flux_central,
            surface_integral=SurfaceIntegralWeakForm(surface_flux),
            volume_integral=VolumeIntegralWeakForm())

Create a discontinuous Galerkin spectral element method (DGSEM) using a
[`LobattoLegendreBasis`](@ref) or a [`GaussLegendreBasis`](@ref) with polynomials of degree `polydeg`.
"""
const DGSEM = DG{Basis} where {Basis <: AbstractBasisSBP}

# The IDP mortar limiting reads the local bounds and the limiter parameters through both
# `dg.mortar.limiter` and `dg.volume_integral.limiter`. These have to be the very same object:
# two separately constructed `SubcellLimiterIDP`s own separate caches, so the bounds written by
# the element limiting would not be the ones read by the mortar limiting.
function check_mortar_limiter(mortar, volume_integral)
    return nothing
end
function check_mortar_limiter(mortar::LobattoLegendreMortarIDP, volume_integral)
    if !(volume_integral isa VolumeIntegralSubcellLimiting)
        throw(ArgumentError("`MortarIDP` requires a `VolumeIntegralSubcellLimiting`, got a `$(typeof(volume_integral))`."))
    end
    if mortar.limiter !== volume_integral.limiter
        throw(ArgumentError("The limiter passed to `MortarIDP` must be the same object as the " *
                            "limiter of the `VolumeIntegralSubcellLimiting`. Construct the " *
                            "`SubcellLimiterIDP` once and pass it to both."))
    end

    return nothing
end

# This API is no longer documented, and we recommend avoiding its public use.
function DGSEM(basis::AbstractBasisSBP,
               surface_flux = flux_central,
               volume_integral = VolumeIntegralWeakForm(),
               mortar = MortarL2(basis))
    check_mortar_limiter(mortar, volume_integral)
    surface_integral = SurfaceIntegralWeakForm(surface_flux)
    return DG{typeof(basis), typeof(mortar), typeof(surface_integral),
              typeof(volume_integral)}(basis, mortar, surface_integral, volume_integral)
end

# This API is no longer documented, and we recommend avoiding its public use.
function DGSEM(basis::AbstractBasisSBP,
               surface_integral::AbstractSurfaceIntegral,
               volume_integral = VolumeIntegralWeakForm(),
               mortar = MortarL2(basis))
    check_mortar_limiter(mortar, volume_integral)
    return DG{typeof(basis), typeof(mortar), typeof(surface_integral),
              typeof(volume_integral)}(basis, mortar, surface_integral, volume_integral)
end

# This API is no longer documented, and we recommend avoiding its public use.
function DGSEM(RealT, polydeg::Integer,
               surface_flux = flux_central,
               volume_integral = VolumeIntegralWeakForm(),
               mortar = MortarL2(LobattoLegendreBasis(RealT, polydeg)))
    basis = LobattoLegendreBasis(RealT, polydeg)

    return DGSEM(basis, surface_flux, volume_integral, mortar)
end

# This API is no longer documented, and we recommend avoiding its public use.
function DGSEM(polydeg::Integer, surface_flux = flux_central,
               volume_integral = VolumeIntegralWeakForm())
    return DGSEM(Float64, polydeg, surface_flux, volume_integral)
end

# The constructor using only keyword arguments is convenient for elixirs since
# it allows to modify the polynomial degree and other parameters via
# `trixi_include`.
function DGSEM(; RealT = Float64,
               polydeg::Integer,
               basis_type = LobattoLegendreBasis,
               surface_flux = flux_central,
               surface_integral = SurfaceIntegralWeakForm(surface_flux),
               volume_integral = VolumeIntegralWeakForm())
    basis = basis_type(RealT, polydeg)
    return DGSEM(basis, surface_integral, volume_integral)
end

@inline polydeg(dg::DGSEM) = polydeg(dg.basis)

Base.summary(io::IO, dg::DGSEM) = print(io, "DGSEM(polydeg=$(polydeg(dg)))")

include("utils_u_mean.jl")

include("containers.jl")

include("indicators.jl")
include("special_volume_integrals.jl")
include("calc_volume_integral.jl")
end # @muladd
