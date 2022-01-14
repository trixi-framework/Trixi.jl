# Note: we define type aliases outside of the @muladd block to avoid Revise breaking when code
# inside the @muladd block is edited. See https://github.com/trixi-framework/Trixi.jl/issues/801
# for more details.

# `DGMulti` refers to both multiple DG types (polynomial/SBP, simplices/quads/hexes) as well as
# the use of multi-dimensional operators in the solver.
const DGMulti{NDIMS, ElemType, ApproxType, SurfaceIntegral, VolumeIntegral} =
  DG{<:RefElemData{NDIMS, ElemType, ApproxType}, Mortar, SurfaceIntegral, VolumeIntegral} where {Mortar}

# Type aliases. The first parameter is `ApproxType` since it is more commonly used for dispatch.
const DGMultiWeakForm{ApproxType, ElemType} =
  DGMulti{NDIMS, ElemType, ApproxType, <:SurfaceIntegralWeakForm, <:VolumeIntegralWeakForm} where {NDIMS}

const DGMultiFluxDiff{ApproxType, ElemType} =
  DGMulti{NDIMS, ElemType, ApproxType, <:SurfaceIntegralWeakForm, <:VolumeIntegralFluxDifferencing} where {NDIMS}

const DGMultiFluxDiffSBP{ApproxType, ElemType} =
  DGMulti{NDIMS, ElemType, ApproxType, <:SurfaceIntegralWeakForm, <:VolumeIntegralFluxDifferencing} where {NDIMS, ApproxType<:Union{SBP, AbstractDerivativeOperator}}

const DGMultiSBP{ApproxType, ElemType} =
  DGMulti{NDIMS, ElemType, ApproxType, SurfaceIntegral, VolumeIntegral} where {NDIMS, ElemType, ApproxType<:Union{SBP, AbstractDerivativeOperator}, SurfaceIntegral, VolumeIntegral}


# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin

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
function DGMulti(; polydeg=nothing,
                   element_type::AbstractElemShape,
                   approximation_type=Polynomial(),
                   surface_flux=flux_central,
                   surface_integral=SurfaceIntegralWeakForm(surface_flux),
                   volume_integral=VolumeIntegralWeakForm(),
                   kwargs...)

  # call dispatchable constructor
  DGMulti(element_type, approximation_type, volume_integral, surface_integral;
          polydeg=polydeg, kwargs...)
end

# dispatchable constructor for DGMulti to allow for specialization
function DGMulti(element_type::AbstractElemShape,
                 approximation_type,
                 volume_integral,
                 surface_integral;
                 polydeg::Integer,
                 kwargs...)

  rd = RefElemData(element_type, approximation_type, polydeg; kwargs...)
  return DG(rd, nothing #= mortar =#, surface_integral, volume_integral)
end

########################################
#            DGMultiMesh
########################################

# now that `DGMulti` is defined, we can define constructors for `DGMultiMesh` which use `dg::DGMulti`

function DGMultiMesh(dg::DGMulti, geometric_term_type, md::MeshData{NDIMS}, boundary_faces) where {NDIMS}
  return DGMultiMesh{NDIMS, geometric_term_type, typeof(md), typeof(boundary_faces)}(md, boundary_faces)
end

# TODO: DGMulti, v0.5. These constructors which use `rd::RefElemData` are now redundant and can be removed.
function DGMultiMesh(vertex_coordinates::NTuple{NDIMS, Vector{Tv}}, EToV::Array{Ti,2}, rd::RefElemData;
                     is_on_boundary = nothing,
                     periodicity=ntuple(_->false, NDIMS), kwargs...) where {NDIMS, Tv, Ti}

  Base.depwarn("`DGMultiMesh` constructor with `rd::RefElemData` is deprecated. Use the constructor with `dg::DGMulti` instead.",
               :DGMultiMesh)
  if haskey(kwargs, :is_periodic)
    # TODO: DGMulti, v0.5. Remove deprecated keyword
    Base.depwarn("keyword argument `is_periodic` is now `periodicity`.", :DGMultiMesh)
    periodicity=kwargs[:is_periodic]
  end

  md = MeshData(vertex_coordinates, EToV, rd)
  if NDIMS==1
    md = StartUpDG.make_periodic(md, periodicity...)
  else
    md = StartUpDG.make_periodic(md, periodicity)
  end
  boundary_faces = StartUpDG.tag_boundary_faces(md, is_on_boundary)
  return DGMultiMesh{NDIMS, typeof(rd.elementType), typeof(md), typeof(boundary_faces)}(md, boundary_faces)
end

function DGMultiMesh(triangulateIO, rd::RefElemData{2, Tri}, boundary_dict::Dict{Symbol, Int})

  vertex_coordinates, EToV = StartUpDG.triangulateIO_to_VXYEToV(triangulateIO)
  md = MeshData(vertex_coordinates, EToV, rd)
  boundary_faces = StartUpDG.tag_boundary_faces(triangulateIO, rd, md, boundary_dict)
  return DGMultiMesh{2, typeof(rd.elementType), typeof(md), typeof(boundary_faces)}(md, boundary_faces)
end

# TODO: DGMulti, v0.5. Remove deprecated constructor
@deprecate VertexMappedMesh(args...; kwargs...) DGMultiMesh(args...; kwargs...)

# Mesh types used internally for trait dispatch
struct Cartesian end
struct VertexMapped end # where element geometry is determined by vertices.
struct Curved end

# type parameters for dispatch using `DGMultiMesh`
abstract type GeometricTermsType end
struct Affine <: GeometricTermsType end # mesh produces constant geometric terms
struct NonAffine <: GeometricTermsType end # mesh produces non-constant geometric terms

# choose MeshType based on the constructor and element type
GeometricTermsType(mesh_type, dg::DGMulti) = GeometricTermsType(mesh_type, dg.basis.elementType)
GeometricTermsType(mesh_type::Cartesian, element_type::AbstractElemShape) = Affine()
GeometricTermsType(mesh_type::TriangulateIO, element_type::Tri) = Affine()
GeometricTermsType(mesh_type::VertexMapped, element_type::Union{Tri, Tet}) = Affine()
GeometricTermsType(mesh_type::VertexMapped, element_type::Union{Quad, Hex}) = NonAffine()
GeometricTermsType(mesh_type::Curved, element_type::AbstractElemShape) = NonAffine()

# other potential constructor types to add later: Bilinear, Isoparametric{polydeg_geo}, Rational/Exact?
# other potential mesh types to add later: Polynomial{polydeg_geo}?

"""
  DGMultiMesh(vertex_coordinates, EToV, dg::DGMulti{NDIMS};
              is_on_boundary=nothing,
              periodicity=ntuple(_->false, NDIMS)) where {NDIMS, Tv}

- `vertex_coordinates` is a tuple of vectors containing x,y,... components of the vertex coordinates
- `EToV` is a 2D array containing element-to-vertex connectivities for each element
- `dg::DGMulti` contains information associated with to the reference element (e.g., quadrature,
  basis evaluation, differentiation, etc).
- `is_on_boundary` specifies boundary using a `Dict{Symbol, <:Function}`
- `periodicity` is a tuple of booleans specifying if the domain is periodic `true`/`false` in the
  (x,y,z) direction.
"""
# TODO: DGMulti v0.5. Standardize order of arguments, pass in `dg` first
function DGMultiMesh(vertex_coordinates, EToV, dg::DGMulti{NDIMS};
                     is_on_boundary=nothing,
                     periodicity=ntuple(_->false, NDIMS), kwargs...) where {NDIMS}
  if haskey(kwargs, :is_periodic)
    # TODO: DGMulti, v0.5. Remove deprecated keyword
    Base.depwarn("keyword argument `is_periodic` is now `periodicity`.", :DGMultiMesh)
    periodicity=kwargs[:is_periodic]
  end

  md = MeshData(vertex_coordinates, EToV, dg.basis)
  if NDIMS == 1
    md = StartUpDG.make_periodic(md, periodicity...)
  else
    md = StartUpDG.make_periodic(md, periodicity)
  end
  boundary_faces = StartUpDG.tag_boundary_faces(md, is_on_boundary)
  return DGMultiMesh(dg, GeometricTermsType(VertexMapped(), dg), md, boundary_faces)
end

"""
    DGMultiMesh(triangulateIO, dg::DGMulti{2, Tri}, boundary_dict::Dict{Symbol, Int})

- `triangulateIO` is a `TriangulateIO` mesh representation
- `dg::DGMulti` contains information associated with to the reference element (e.g., quadrature,
  basis evaluation, differentiation, etc).
- `boundary_dict` is a `Dict{Symbol, Int}` which associates each integer `TriangulateIO` boundary
  tag with a `Symbol`.
"""
# TODO: DGMulti v0.5, standardize order of arguments (`dg` first)
function DGMultiMesh(triangulateIO, dg::DGMulti{2, Tri}, boundary_dict::Dict{Symbol, Int};
                     periodicity=(false, false))
  vertex_coordinates, EToV = StartUpDG.triangulateIO_to_VXYEToV(triangulateIO)
  md = MeshData(vertex_coordinates, EToV, dg.basis)
  md = StartUpDG.make_periodic(md, periodicity)
  boundary_faces = StartUpDG.tag_boundary_faces(triangulateIO, dg.basis, md, boundary_dict)
  return DGMultiMesh(dg, GeometricTermsType(TriangulateIO(), dg), md, boundary_faces)
end

# TODO: DGMulti. Make `cells_per_dimension` a non-keyword argument for easier dispatch.
"""
    DGMultiMesh(dg::DGMulti; cells_per_dimension,
                coordinates_min=(-1.0, -1.0), coordinates_max=(1.0, 1.0),
                is_on_boundary=nothing,
                periodicity=ntuple(_ -> false, NDIMS))

Constructs a Cartesian [`DGMultiMesh`](@ref) with element type `dg.basis.elementType`. The domain is
the tensor product of the intervals `[coordinates_min[i], coordinates_max[i]]`.
- `is_on_boundary` specifies boundary using a `Dict{Symbol, <:Function}`
- `periodicity` is a tuple of `Bool`s specifying periodicity = `true`/`false` in the (x,y,z) direction.
"""
function DGMultiMesh(dg::DGMulti{NDIMS}; cells_per_dimension,
                     coordinates_min=ntuple(_ -> -one(real(dg)), NDIMS),
                     coordinates_max=ntuple(_ -> one(real(dg)), NDIMS),
                     is_on_boundary=nothing,
                     periodicity=ntuple(_ -> false, NDIMS), kwargs...) where {NDIMS}

  if haskey(kwargs, :is_periodic)
    # TODO: DGMulti. Deprecate `is_periodic` in version 0.5
    Base.depwarn("keyword argument `is_periodic` is now `periodicity`.", :DGMultiMesh)
    periodicity=kwargs[:is_periodic]
  end

  vertex_coordinates, EToV = StartUpDG.uniform_mesh(dg.basis.elementType, cells_per_dimension...)
  domain_lengths = coordinates_max .- coordinates_min
  for i in 1:NDIMS
    @. vertex_coordinates[i] = 0.5 * (vertex_coordinates[i] + 1) * domain_lengths[i] + coordinates_min[i]
  end

  md = MeshData(vertex_coordinates, EToV, dg.basis)
  if NDIMS == 1
    md = StartUpDG.make_periodic(md, periodicity...)
  else
    md = StartUpDG.make_periodic(md, periodicity)
  end
  boundary_faces = StartUpDG.tag_boundary_faces(md, is_on_boundary)
  return DGMultiMesh(dg, GeometricTermsType(Cartesian(), dg), md, boundary_faces)
end

"""
    DGMultiMesh(dg::DGMulti{NDIMS}, cells_per_dimension, mapping;
                is_on_boundary=nothing,
                periodicity=ntuple(_ -> false, NDIMS), kwargs...) where {NDIMS}

Constructs a `Curved()` [`DGMultiMesh`](@ref) with element type `dg.basis.elementType`.
- `mapping` is a function which maps from a reference [-1, 1]^NDIMS domain to a mapped domain,
   e.g., `xy = mapping(x, y)` in 2D.
- `is_on_boundary` specifies boundary using a `Dict{Symbol, <:Function}`
- `periodicity` is a tuple of `Bool`s specifying periodicity = `true`/`false` in the (x,y,z) direction.
"""
function DGMultiMesh(dg::DGMulti{NDIMS}, cells_per_dimension, mapping;
                     is_on_boundary=nothing,
                     periodicity=ntuple(_ -> false, NDIMS), kwargs...) where {NDIMS}

  vertex_coordinates, EToV = StartUpDG.uniform_mesh(dg.basis.elementType, cells_per_dimension...)
  md = MeshData(vertex_coordinates, EToV, dg.basis)
  md = NDIMS==1 ? StartUpDG.make_periodic(md, periodicity...) : StartUpDG.make_periodic(md, periodicity)

  @unpack xyz = md
  for i in eachindex(xyz[1])
    new_xyz = mapping(getindex.(xyz, i)...)
    setindex!.(xyz, new_xyz, i)
  end
  md_curved = MeshData(dg.basis, md, xyz...)

  # interpolate geometric terms to both volume and face cubature points
  @unpack rstxyzJ = md_curved
  @unpack Vq, Vf = dg.basis
  rstxyzJ_interpolated = map(x -> [Vq; Vf] * x, rstxyzJ)
  md_curved = @set md_curved.rstxyzJ = rstxyzJ_interpolated

  boundary_faces = StartUpDG.tag_boundary_faces(md_curved, is_on_boundary)
  return DGMultiMesh(dg, GeometricTermsType(Curved(), dg), md_curved, boundary_faces)
end

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

# `SimpleKronecker` lazily stores a Kronecker product `kron(ntuple(A, NDIMS)...)`.
# This object also allocates some temporary storage to enable the fast computation
# of matrix-vector products.
struct SimpleKronecker{NDIMS, TA, Ttmp}
  A::TA
  tmp_storage::Ttmp # temporary array used for Kronecker multiplication
end

# constructor for SimpleKronecker which requires specifying only `NDIMS` and
# the 1D matrix `A`.
function SimpleKronecker(NDIMS, A, eltype_A=eltype(A))
  @assert size(A, 1) == size(A, 2) # check if square
  tmp_storage=[zeros(eltype_A, ntuple(_ -> size(A, 2), NDIMS)...) for _ in 1:Threads.nthreads()]
  return SimpleKronecker{NDIMS, typeof(A), typeof(tmp_storage)}(A, tmp_storage)
end

# Computes `b = kron(A, A) * x` in an optimized fashion
function LinearAlgebra.mul!(b_in, A_kronecker::SimpleKronecker{2}, x_in)

  @unpack A = A_kronecker
  tmp_storage = A_kronecker.tmp_storage[Threads.threadid()]
  n = size(A, 2)

  # copy `x_in` to `tmp_storage` to avoid mutating the input
  @assert length(tmp_storage) == length(x_in)
  for i in eachindex(tmp_storage)
    tmp_storage[i] = x_in[i]
  end
  x = reshape(tmp_storage, n, n)
  b = reshape(b_in, n, n)

  @turbo for j in 1:n, i in 1:n
    tmp = zero(eltype(x))
    for ii in 1:n
      tmp = tmp + A[i, ii] * x[ii, j]
    end
    b[i, j] = tmp
  end

  @turbo for j in 1:n, i in 1:n
    tmp = zero(eltype(x))
    for jj in 1:n
      tmp = tmp + A[j, jj] * b[i, jj]
    end
    x[i, j] = tmp
  end

  @turbo for i in eachindex(b_in)
    b_in[i] = x[i]
  end

  return nothing
end

# Computes `b = kron(A, A, A) * x` in an optimized fashion
function LinearAlgebra.mul!(b_in, A_kronecker::SimpleKronecker{3}, x_in)

  @unpack A = A_kronecker
  tmp_storage = A_kronecker.tmp_storage[Threads.threadid()]
  n = size(A, 2)

  # copy `x_in` to `tmp_storage` to avoid mutating the input
  for i in eachindex(tmp_storage)
    tmp_storage[i] = x_in[i]
  end
  x = reshape(tmp_storage, n, n, n)
  b = reshape(b_in, n, n, n)

  @turbo for k in 1:n, j in 1:n, i in 1:n
    tmp = zero(eltype(x))
    for ii in 1:n
      tmp = tmp + A[i, ii] * x[ii, j, k]
    end
    b[i, j, k] = tmp
  end

  @turbo for k in 1:n, j in 1:n, i in 1:n
    tmp = zero(eltype(x))
    for jj in 1:n
      tmp = tmp + A[j, jj] * b[i, jj, k]
    end
    x[i, j, k] = tmp
  end

  @turbo for k in 1:n, j in 1:n, i in 1:n
    tmp = zero(eltype(x))
    for kk in 1:n
      tmp = tmp + A[k, kk] * x[i, j, kk]
    end
    b[i, j, k] = tmp
  end

  return nothing
end



end # @muladd
