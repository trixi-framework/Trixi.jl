
# `DGMulti` refers to both multiple DG types (polynomial/SBP, simplices/quads/hexes) as well as
# the use of multi-dimensional operators in the solver.
const DGMulti{NDIMS, ElemType, ApproxType, SurfaceIntegral, VolumeIntegral} =
  DG{<:RefElemData{NDIMS, ElemType, ApproxType}, Mortar, SurfaceIntegral, VolumeIntegral} where {Mortar}

# these are necessary for pretty printing
polydeg(dg::DGMulti) = dg.basis.N
Base.summary(io::IO, dg::DG) where {DG <: DGMulti} = print(io, "DGMulti(polydeg=$(polydeg(dg)))")
Base.real(rd::RefElemData{NDIMS, Elem, ApproxType, Nfaces, RealT}) where {NDIMS, Elem, ApproxType, Nfaces, RealT} = RealT

"""
    DGMulti(; polydeg::Integer,
              element_type::AbstractElemShape,
              approximation_type=Polynomial(),
              surface_flux=flux_central,
              surface_integral=SurfaceIntegralWeakForm(surface_flux),
              volume_integral=VolumeIntegralWeakForm())

Create a discontinuous Galerkin method which uses
- approximations of polynomial degree `polydeg`
- element type `element_type` (`Tri()`, `Quad()`, `Tet()`, and `Hex()` currently supported)

Optional:
- `approximation_type` (default is `Polynomial()`; `SBP()` also supported for `Tri()`, `Quad()`,
  and `Hex()` element types).
"""
function DGMulti(; polydeg::Integer,
                   element_type::AbstractElemShape,
                   approximation_type=Polynomial(),
                   surface_flux=flux_central,
                   surface_integral=SurfaceIntegralWeakForm(surface_flux),
                   volume_integral=VolumeIntegralWeakForm())
  rd = RefElemData(element_type, approximation_type, polydeg)
  return DG(rd, nothing #= mortar =#, surface_integral, volume_integral)
end

# Type aliases. The first parameter is `ApproxType` since it is more commonly used for dispatch.
const DGMultiWeakForm{ApproxType, ElemType} =
  DGMulti{NDIMS, ElemType, ApproxType, <:SurfaceIntegralWeakForm, <:VolumeIntegralWeakForm} where {NDIMS}

const DGMultiFluxDiff{ApproxType, ElemType} =
  DGMulti{NDIMS, ElemType, ApproxType, <:SurfaceIntegralWeakForm, <:VolumeIntegralFluxDifferencing} where {NDIMS}

# Todo: simplices. Add traits for dispatch on affine/curved meshes here.
