
# `MultiDG` refers to both multiple DG types (polynomial/SBP, simplices/quads/hexes) as well as
# the use of multi-dimensional operators in the solver.
const MultiDG{NDIMS, ElemType, ApproxType, SurfaceIntegral, VolumeIntegral} =
  DG{<:RefElemData{NDIMS, ElemType, ApproxType}, Mortar, SurfaceIntegral, VolumeIntegral} where {Mortar}

"""
    MultiDG(; polydeg::Integer,
              elem_type::AbstractElemShape,
              approximation_type=Polynomial(),
              surface_flux=flux_central,
              surface_integral=SurfaceIntegralWeakForm(surface_flux),
              volume_integral=VolumeIntegralWeakForm())

Create a discontinuous Galerkin method which uses
- approximations of polynomial degree `polydeg`
- element type `elem_type` (`Tri()`, `Quad()`, `Tet()`, and `Hex()` currently supported)

Optional:
- approximation type of `approximation_type` (default is `Polynomial()`; `SBP()` also supported for
`Tri()`, `Quad()`, and `Hex()` element types).
"""
function MultiDG(; polydeg::Integer,
                   elem_type::AbstractElemShape,
                   approximation_type=Polynomial(),
                   surface_flux=flux_central,
                   surface_integral=SurfaceIntegralWeakForm(surface_flux),
                   volume_integral=VolumeIntegralWeakForm())
  rd = RefElemData(elem_type, approximation_type, polydeg)
  return MultiDG(rd, surface_integral, volume_integral)
end

# type aliases for dispatch purposes
const MultiDGWeakForm{NDIMS, ElemType, ApproxType} =
  MultiDG{NDIMS, ElemType, ApproxType, <:SurfaceIntegralWeakForm, <:VolumeIntegralWeakForm}

const PolyDGFluxDiff{NDIMS, ElemType} =
  MultiDG{NDIMS, ElemType, Polynomial, <:SurfaceIntegralWeakForm, <:VolumeIntegralFluxDifferencing} where {NDIMS, ElemType}

const SBPDGFluxDiff{Dim, Elem} =
  MultiDG{NDIMS, ElemType, <:SBP, <:SurfaceIntegralWeakForm, <:VolumeIntegralFluxDifferencing} where {NDIMS, ElemType}


# Todo: simplices. Add traits for dispatch on affine/curved meshes here.