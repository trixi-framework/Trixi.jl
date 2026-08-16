# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

"""
    DirectStiffnessSummation()

Coupling of the elements of the [`CGSEM`](@ref). Since the numerical solution is
continuous across element interfaces, the surface contributions of two neighboring
elements cancel each other. Thus, there is neither a surface integral nor a
numerical surface flux. Instead, the element-local contributions of the degrees of
freedom shared by neighboring elements are combined by a direct stiffness
summation (also known as gather-scatter operation).
"""
struct DirectStiffnessSummation <: AbstractSurfaceIntegral end

"""
    CGSEM(; RealT = Float64,
            polydeg::Integer,
            volume_integral = VolumeIntegralFluxDifferencing(flux_central))

Create a continuous Galerkin spectral element method (CGSEM) using a
[`LobattoLegendreBasis`](@ref) with polynomials of degree `polydeg`.

The CGSEM uses the same element-local flux differencing volume integral as the
[`DGSEM`](@ref), see [`VolumeIntegralFluxDifferencing`](@ref). The elements are
coupled by a [`DirectStiffnessSummation`](@ref). The resulting global operator
inherits the summation-by-parts property of the element-local operators. Thus,
the CGSEM is entropy conservative if the `volume_flux` is entropy conservative.
Note that it does not contain any interface dissipation.

Currently implemented for conforming, periodic `TreeMesh{2}`es.

## References

- Kopriva (2009)
  Implementing Spectral Methods for Partial Differential Equations:
  Algorithms for Scientists and Engineers
  [doi: 10.1007/978-90-481-2261-5](https://doi.org/10.1007/978-90-481-2261-5)
"""
const CGSEM = DG{Basis, Nothing, DirectStiffnessSummation,
                 VolumeIntegral} where {Basis <: LobattoLegendreBasis, VolumeIntegral}

function CGSEM(basis::LobattoLegendreBasis,
               volume_integral = VolumeIntegralFluxDifferencing(flux_central))
    # The CGSEM supports only conforming meshes and thus does not use mortars.
    mortar = nothing
    surface_integral = DirectStiffnessSummation()

    return DG{typeof(basis), typeof(mortar), typeof(surface_integral),
              typeof(volume_integral)}(basis, mortar, surface_integral, volume_integral)
end

# The constructor using only keyword arguments is convenient for elixirs since
# it allows to modify the polynomial degree and other parameters via
# `trixi_include`.
function CGSEM(; RealT = Float64,
               polydeg::Integer,
               volume_integral = VolumeIntegralFluxDifferencing(flux_central))
    basis = LobattoLegendreBasis(RealT, polydeg)
    return CGSEM(basis, volume_integral)
end

function Base.show(io::IO, cg::CGSEM)
    @nospecialize cg # reduce precompilation time

    print(io, "CGSEM{", real(cg), "}(")
    print(io, cg.basis)
    print(io, ", ", cg.volume_integral)
    print(io, ")")
    return nothing
end

function Base.show(io::IO, mime::MIME"text/plain", cg::CGSEM)
    @nospecialize cg # reduce precompilation time

    if get(io, :compact, false)
        show(io, cg)
    else
        summary_header(io, "CGSEM{" * string(real(cg)) * "}")
        summary_line(io, "basis", cg.basis)
        summary_line(io, "element coupling", cg.surface_integral |> typeof |> nameof)
        summary_line(io, "volume integral", cg.volume_integral |> typeof |> nameof)
        show(increment_indent(io), mime, cg.volume_integral)
        summary_footer(io)
    end
end

Base.summary(io::IO, cg::CGSEM) = print(io, "CGSEM(polydeg=$(polydeg(cg)))")
end # @muladd
