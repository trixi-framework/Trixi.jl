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
    DGSEM(; RealT=Float64, polydeg::Integer,
            basis = LobattoLegendreBasis(RealT, polydeg)
            surface_flux=flux_central,
            surface_integral=SurfaceIntegralWeakForm(surface_flux),
            volume_integral=VolumeIntegralWeakForm())

Create a discontinuous Galerkin spectral element method (DGSEM) using a
[`LobattoLegendreBasis`](@ref) with polynomials of degree `polydeg`.
"""
const DGSEM = DG{Basis} where {Basis <: AbstractBasisSBP}

# This API is no longer documented, and we recommend avoiding its public use.
function DGSEM(basis::LobattoLegendreBasis,
               surface_flux = flux_central,
               volume_integral = VolumeIntegralWeakForm(),
               mortar = MortarL2(basis))
    surface_integral = SurfaceIntegralWeakForm(surface_flux)
    return DG{typeof(basis), typeof(mortar), typeof(surface_integral),
              typeof(volume_integral)}(basis, mortar, surface_integral, volume_integral)
end

# This API is no longer documented, and we recommend avoiding its public use.
function DGSEM(basis::LobattoLegendreBasis,
               surface_integral::AbstractSurfaceIntegral,
               volume_integral = VolumeIntegralWeakForm(),
               mortar = MortarL2(basis))
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
               basis = LobattoLegendreBasis(RealT, polydeg),
               surface_flux = flux_central,
               surface_integral = SurfaceIntegralWeakForm(surface_flux),
               volume_integral = VolumeIntegralWeakForm())
    return DGSEM(basis, surface_integral, volume_integral)
end

@inline polydeg(dg::DGSEM) = polydeg(dg.basis)

Base.summary(io::IO, dg::DGSEM) = print(io, "DGSEM(polydeg=$(polydeg(dg)))")

include("compute_u_mean.jl")
include("containers.jl")
include("indicators.jl")
include("calc_volume_integral.jl")
end # @muladd
