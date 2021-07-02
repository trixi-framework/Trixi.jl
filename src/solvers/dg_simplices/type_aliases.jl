const DGWeakForm{Dims, ElemType} = DG{<:RefElemData{Dims, ElemType}, Mortar, 
                    <:SurfaceIntegralWeakForm,
                    <:VolumeIntegralWeakForm} where {Mortar}
const DGFluxDiff{Dim, Elem} = DG{<:RefElemData{Dim, Elem}, 
                            M, <:SurfaceIntegralWeakForm, <:VolumeIntegralFluxDifferencing} where {Dim, Elem, M}
const PolyDGFluxDiff{Dim, Elem} = DG{<:RefElemData{Dim, Elem, Polynomial}, 
                            M, <:SurfaceIntegralWeakForm, <:VolumeIntegralFluxDifferencing} where {Dim, Elem, M}
const SBPDGFluxDiff{Dim, Elem} = DG{<:RefElemData{Dim, Elem, <:SBP}, 
                            M, <:SurfaceIntegralWeakForm, <:VolumeIntegralFluxDifferencing} where {Dim, Elem, M}
                    