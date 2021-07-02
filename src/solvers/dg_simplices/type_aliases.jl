const DGWeakForm{Dims, ElemType} = DG{<:RefElemData{Dims, ElemType}, Mortar, 
                    <:SurfaceIntegralWeakForm,
                    <:VolumeIntegralWeakForm} where {Mortar}
const DGFluxDiff{Dim, Elem} = DG{<:RefElemData{Dim, Elem}, 
                            M, S, <: VolumeIntegralFluxDifferencing} where {Dim, Elem, M, S}
const PolyDGFluxDiff{Dim, Elem} = DG{<:RefElemData{Dim, Elem, Polynomial}, 
                            M, S, <: VolumeIntegralFluxDifferencing} where {Dim, Elem, M, S}
const SBPDGFluxDiff{Dim, Elem} = DG{<:RefElemData{Dim, Elem, <:SBP}, 
                            M, S, <: VolumeIntegralFluxDifferencing} where {Dim, Elem, M, S}
                    