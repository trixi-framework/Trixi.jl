
# `MultiDG` refers to both multiple DG types (polynomial/SBP, simplices/quads/hexes) as well as
# the use of multi-dimensional operators in the solver.
const MultiDG{NDIMS, ElemType, ApproxType, SurfaceIntegral, VolumeIntegral} =
  DG{<:RefElemData{NDIMS, ElemType, ApproxType}, Mortar, SurfaceIntegral, VolumeIntegral} where {Mortar}

const MultiDGWeakForm{NDIMS, ElemType, ApproxType} =
  MultiDG{NDIMS, ElemType, ApproxType, <:SurfaceIntegralWeakForm, <:VolumeIntegralWeakForm}

const PolyDGFluxDiff{NDIMS, ElemType} =
  MultiDG{NDIMS, ElemType, Polynomial, <:SurfaceIntegralWeakForm, <:VolumeIntegralFluxDifferencing} where {NDIMS, ElemType}

const SBPDGFluxDiff{Dim, Elem} =
  MultiDG{NDIMS, ElemType, <:SBP, <:SurfaceIntegralWeakForm, <:VolumeIntegralFluxDifferencing} where {NDIMS, ElemType}


# Todo: simplices. Add traits for dispatch on affine/curved meshes here.