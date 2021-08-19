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
                 volume_integral=VolumeIntegralWeakForm(),
                 surface_integral=SurfaceIntegralWeakForm(surface_flux);
                 polydeg::Integer,
                 surface_flux=flux_central,
                 kwargs...)

  rd = RefElemData(element_type, approximation_type, polydeg, kwargs...)
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

# lazy linear combination of B = ∑_i coeffs[i] * A[i]
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

# ========= GSBP approximation types ============

# GSBP ApproximationType: e.g., Gauss nodes on quads/hexes
struct GSBP end

# Todo: DGMulti. Decide if we should add GSBP on triangles.

# Specialized constructor for GSBP approximation type on quad elements. Restricting to
# VolumeKernelFluxDifferencing for now since there isn't a way to exploit this structure for
# VolumeIntegralWeakForm yet.
function DGMulti(element_type::Quad,
                 approximation_type::GSBP,
                 volume_integral::VolumeIntegralFluxDifferencing,
                 surface_integral=SurfaceIntegralWeakForm(surface_flux);
                 polydeg::Integer,
                 surface_flux=flux_central,
                 kwargs...)

  # create tensor product Gauss quadrature rule with polydeg+1 points
  r1D, w1D = StartUpDG.gauss_quad(0, 0, polydeg)
  rq, sq = vec.(StartUpDG.NodesAndModes.meshgrid(r1D))
  wr, ws = vec.(StartUpDG.NodesAndModes.meshgrid(w1D))
  wq = wr .* ws
  gauss_rule_vol = (rq, sq, wq)

  # Gauss quadrature rule on reference face [-1, 1]
  gauss_rule_face = (r1D, w1D)

  rd = RefElemData(element_type, Polynomial(), polydeg,
                   quad_rule_vol=gauss_rule_vol,
                   quad_rule_face=gauss_rule_face,
                   kwargs...)

  # WARNING: somewhat hacky. Since there is no dedicated GSBP approximation type implemented
  # in StartUpDG, we simply initialize `rd = RefElemData(...)` with the appropriate quadrature
  # rules and modify the rd.approximationType manually so we can dispatch on the `GSBP` type.
  # This uses the Setfield @set macro, which aims to do something similar to Trixi.remake.
  rd_gauss = @set rd.approximationType = GSBP()

  # We modify the projection operator so that we can reuse `compute_coefficients!` since it
  # assigns solution values at quadrature points (e.g., Gauss points), then projects the result.
  rd_gauss = @set rd_gauss.Pq = UniformScaling{Bool}(true) # equivalent to LinearAlgebra.I

  # We will modify the face interpolation operator of rd_gauss later, but want to do so only after
  # the mesh is initialized, since the face interpolation operator is used for that.
  return DG(rd_gauss, nothing #= mortar =#, surface_integral, volume_integral)
end


end # @muladd
