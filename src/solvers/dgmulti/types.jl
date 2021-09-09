# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin

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
              volume_integral=VolumeIntegralWeakForm(),
              RefElemData_kwargs...)

Create a discontinuous Galerkin method which uses
- approximations of polynomial degree `polydeg`
- element type `element_type` (`Tri()`, `Quad()`, `Tet()`, and `Hex()` currently supported)

Optional:
- `approximation_type` (default is `Polynomial()`; `SBP()` also supported for `Tri()`, `Quad()`,
  and `Hex()` element types).
- `RefElemData_kwargs` are additional keyword arguments for `RefElemData`, such as `quad_rule_vol`.
  For more info, see the [StartUpDG.jl docs](https://jlchan.github.io/StartUpDG.jl/dev/).
"""
function DGMulti(; polydeg::Integer,
                   element_type::AbstractElemShape,
                   approximation_type=Polynomial(),
                   surface_flux=flux_central,
                   surface_integral=SurfaceIntegralWeakForm(surface_flux),
                   volume_integral=VolumeIntegralWeakForm(),
                   kwargs...)

  # call dispatchable constructor
  DGMulti(element_type, approximation_type, volume_integral, surface_integral;
          polydeg=polydeg, surface_flux=surface_flux, kwargs...)
end

# dispatchable constructor for DGMulti to allow for specialization
function DGMulti(element_type::AbstractElemShape,
                 approximation_type,
                 volume_integral,
                 surface_integral;
                 polydeg::Integer,
                 surface_flux,
                 kwargs...)

  rd = RefElemData(element_type, approximation_type, polydeg; kwargs...)
  return DG(rd, nothing #= mortar =#, surface_integral, volume_integral)
end

# Type aliases. The first parameter is `ApproxType` since it is more commonly used for dispatch.
const DGMultiWeakForm{ApproxType, ElemType} =
  DGMulti{NDIMS, ElemType, ApproxType, <:SurfaceIntegralWeakForm, <:VolumeIntegralWeakForm} where {NDIMS}

const DGMultiFluxDiff{ApproxType, ElemType} =
  DGMulti{NDIMS, ElemType, ApproxType, <:SurfaceIntegralWeakForm, <:VolumeIntegralFluxDifferencing} where {NDIMS}


# now that DGMulti is defined, we can define constructors for VertexMappedMesh which use dg::DGMulti
"""
    VertexMappedMesh(vertex_coordinates, EToV, dg::DGMulti;
                     is_on_boundary = nothing,
                     is_periodic::NTuple{NDIMS, Bool} = ntuple(_->false, NDIMS)) where {NDIMS, Tv}

Constructor which uses `dg::DGMulti` instead of `rd::RefElemData`.
"""
VertexMappedMesh(vertex_coordinates, EToV, dg::DGMulti; kwargs...) =
  VertexMappedMesh(vertex_coordinates, EToV, dg.basis; kwargs...)

"""
    VertexMappedMesh(triangulateIO, dg::DGMulti, boundary_dict::Dict{Symbol, Int})

Constructor which uses `dg::DGMulti` instead of `rd::RefElemData`.
"""
VertexMappedMesh(triangulateIO, dg::DGMulti, boundary_dict::Dict{Symbol, Int}) =
  VertexMappedMesh(triangulateIO, dg.basis, boundary_dict)

# Todo: DGMulti. Add traits for dispatch on affine/curved meshes here.

# Matrix type for lazy construction of physical differentiation matrices
# Constructs a lazy linear combination of B = ∑_i coeffs[i] * A[i]
struct LazyMatrixLinearCombo{Tcoeffs, N, Tv, TA <: AbstractMatrix{Tv}} <: AbstractMatrix{Tv}
  matrices::NTuple{N, TA}
  coeffs::NTuple{N, Tcoeffs}
  function LazyMatrixLinearCombo(matrices, coeffs)
    @assert all(matrix -> size(matrix) == size(first(matrices)), matrices)
    new{typeof(first(coeffs)), length(matrices), eltype(first(matrices)), typeof(first(matrices))}(matrices, coeffs)
  end
end
Base.eltype(A::LazyMatrixLinearCombo) = eltype(first(A.matrices))
Base.IndexStyle(A::LazyMatrixLinearCombo) = IndexCartesian()
Base.size(A::LazyMatrixLinearCombo) = size(first(A.matrices))

@inline function Base.getindex(A::LazyMatrixLinearCombo{<:Real, N}, i, j) where {N}
  val = zero(eltype(A))
  for k in Base.OneTo(N)
    val = val + A.coeffs[k] * getindex(A.matrices[k], i, j)
  end
  return val
end


end # @muladd
