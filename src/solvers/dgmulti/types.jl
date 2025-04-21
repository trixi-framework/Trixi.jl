# Note: we define type aliases outside of the @muladd block to avoid Revise breaking when code
# inside the @muladd block is edited. See https://github.com/trixi-framework/Trixi.jl/issues/801
# for more details.

# `DGMulti` refers to both multiple DG types (polynomial/SBP, simplices/quads/hexes) as well as
# the use of multi-dimensional operators in the solver.
const DGMulti{NDIMS, ElemType, ApproxType, SurfaceIntegral, VolumeIntegral} = DG{<:RefElemData{NDIMS,
                                                                                               ElemType,
                                                                                               ApproxType},
                                                                                 Mortar,
                                                                                 SurfaceIntegral,
                                                                                 VolumeIntegral} where {
                                                                                                        Mortar
                                                                                                        }

# Type aliases. The first parameter is `ApproxType` since it is more commonly used for dispatch.
const DGMultiWeakForm{ApproxType, ElemType} = DGMulti{NDIMS, ElemType, ApproxType,
                                                      <:SurfaceIntegralWeakForm,
                                                      <:VolumeIntegralWeakForm} where {NDIMS
                                                                                       }

const DGMultiFluxDiff{ApproxType, ElemType} = DGMulti{NDIMS, ElemType, ApproxType,
                                                      <:SurfaceIntegralWeakForm,
                                                      <:Union{VolumeIntegralFluxDifferencing,
                                                              VolumeIntegralShockCapturingHG}} where {
                                                                                                      NDIMS
                                                                                                      }

const DGMultiFluxDiffSBP{ApproxType, ElemType} = DGMulti{NDIMS, ElemType, ApproxType,
                                                         <:SurfaceIntegralWeakForm,
                                                         <:Union{VolumeIntegralFluxDifferencing,
                                                                 VolumeIntegralShockCapturingHG}} where {
                                                                                                         NDIMS,
                                                                                                         ApproxType <:
                                                                                                         Union{SBP,
                                                                                                               AbstractDerivativeOperator}
                                                                                                         }

const DGMultiSBP{ApproxType, ElemType} = DGMulti{NDIMS, ElemType, ApproxType,
                                                 SurfaceIntegral,
                                                 VolumeIntegral} where {NDIMS, ElemType,
                                                                        ApproxType <:
                                                                        Union{SBP,
                                                                              AbstractDerivativeOperator},
                                                                        SurfaceIntegral,
                                                                        VolumeIntegral}

# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

# these are necessary for pretty printing
polydeg(dg::DGMulti) = dg.basis.N
function Base.summary(io::IO, dg::DG) where {DG <: DGMulti}
    print(io, "DGMulti(polydeg=$(polydeg(dg)))")
end

# real(rd) is the eltype of the nodes `rd.r`.
Base.real(rd::RefElemData) = eltype(rd.r)

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
function DGMulti(; polydeg = nothing,
                 element_type::AbstractElemShape,
                 approximation_type = Polynomial(),
                 surface_flux = flux_central,
                 surface_integral = SurfaceIntegralWeakForm(surface_flux),
                 volume_integral = VolumeIntegralWeakForm(),
                 kwargs...)

    # call dispatchable constructor
    DGMulti(element_type, approximation_type, volume_integral, surface_integral;
            polydeg = polydeg, kwargs...)
end

# dispatchable constructor for DGMulti using a TensorProductWedge
function DGMulti(element_type::Wedge,
                 approximation_type,
                 volume_integral,
                 surface_integral;
                 polydeg::Tuple,
                 kwargs...)
    factor_a = RefElemData(Tri(), approximation_type, polydeg[1]; kwargs...)
    factor_b = RefElemData(Line(), approximation_type, polydeg[2]; kwargs...)

    tensor = TensorProductWedge(factor_a, factor_b)
    rd = RefElemData(element_type, tensor; kwargs...)
    return DG(rd, nothing, surface_integral, volume_integral)
end

# dispatchable constructor for DGMulti to allow for specialization
function DGMulti(element_type::AbstractElemShape,
                 approximation_type,
                 volume_integral,
                 surface_integral;
                 polydeg::Integer,
                 kwargs...)
    rd = RefElemData(element_type, approximation_type, polydeg; kwargs...)
    # `nothing` is passed as `mortar`
    return DG(rd, nothing, surface_integral, volume_integral)
end

function DGMulti(basis::RefElemData; volume_integral, surface_integral)
    # `nothing` is passed as `mortar`
    DG(basis, nothing, surface_integral, volume_integral)
end

"""
    DGMultiBasis(element_type, polydeg; approximation_type = Polynomial(), kwargs...)

Constructs a basis for DGMulti solvers. Returns a "StartUpDG.RefElemData" object.
  The `kwargs` arguments are additional keyword arguments for `RefElemData`, such as `quad_rule_vol`.
  These are the same as the `RefElemData_kwargs` used in [`DGMulti`](@ref).
  For more info, see the [StartUpDG.jl docs](https://jlchan.github.io/StartUpDG.jl/dev/).

"""
function DGMultiBasis(element_type, polydeg; approximation_type = Polynomial(),
                      kwargs...)
    RefElemData(element_type, approximation_type, polydeg; kwargs...)
end

########################################
#            DGMultiMesh
########################################

# now that `DGMulti` is defined, we can define constructors for `DGMultiMesh` which use `dg::DGMulti`

function DGMultiMesh(dg::DGMulti, geometric_term_type, md::MeshData{NDIMS},
                     boundary_faces) where {NDIMS}
    return DGMultiMesh{NDIMS, typeof(geometric_term_type), typeof(md),
                       typeof(boundary_faces)}(md, boundary_faces)
end

# Mesh types used internally for trait dispatch
struct Cartesian end
struct VertexMapped end # where element geometry is determined by vertices.
struct Curved end

# type parameters for dispatch using `DGMultiMesh`
abstract type GeometricTermsType end
struct Affine <: GeometricTermsType end # mesh produces constant geometric terms
struct NonAffine <: GeometricTermsType end # mesh produces non-constant geometric terms

# choose MeshType based on the constructor and element type
function GeometricTermsType(mesh_type, dg::DGMulti)
    GeometricTermsType(mesh_type, dg.basis.element_type)
end
GeometricTermsType(mesh_type::Cartesian, element_type::AbstractElemShape) = Affine()
GeometricTermsType(mesh_type::TriangulateIO, element_type::Tri) = Affine()
GeometricTermsType(mesh_type::VertexMapped, element_type::Union{Tri, Tet}) = Affine()
function GeometricTermsType(mesh_type::VertexMapped, element_type::Union{Quad, Hex})
    NonAffine()
end
GeometricTermsType(mesh_type::Curved, element_type::AbstractElemShape) = NonAffine()

# other potential constructor types to add later: Bilinear, Isoparametric{polydeg_geo}, Rational/Exact?
# other potential mesh types to add later: Polynomial{polydeg_geo}?

"""
    DGMultiMesh(dg::DGMulti{NDIMS}, vertex_coordinates, EToV;
                is_on_boundary=nothing,
                periodicity=ntuple(_->false, NDIMS)) where {NDIMS}

- `dg::DGMulti` contains information associated with to the reference element (e.g., quadrature,
  basis evaluation, differentiation, etc).
- `vertex_coordinates` is a tuple of vectors containing x,y,... components of the vertex coordinates
- `EToV` is a 2D array containing element-to-vertex connectivities for each element
- `is_on_boundary` specifies boundary using a `Dict{Symbol, <:Function}`
- `periodicity` is a tuple of booleans specifying if the domain is periodic `true`/`false` in the
  (x,y,z) direction.
"""
function DGMultiMesh(dg::DGMulti{NDIMS}, vertex_coordinates, EToV::AbstractArray;
                     is_on_boundary = nothing,
                     periodicity = ntuple(_ -> false, NDIMS), kwargs...) where {NDIMS}
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
    DGMultiMesh(dg::DGMulti{2, Tri}, triangulateIO, boundary_dict::Dict{Symbol, Int})

- `dg::DGMulti` contains information associated with to the reference element (e.g., quadrature,
  basis evaluation, differentiation, etc).
- `triangulateIO` is a `TriangulateIO` mesh representation
- `boundary_dict` is a `Dict{Symbol, Int}` which associates each integer `TriangulateIO` boundary
  tag with a `Symbol`.
"""
function DGMultiMesh(dg::DGMulti{2, Tri}, triangulateIO,
                     boundary_dict::Dict{Symbol, Int};
                     periodicity = (false, false))
    vertex_coordinates, EToV = StartUpDG.triangulateIO_to_VXYEToV(triangulateIO)
    md = MeshData(vertex_coordinates, EToV, dg.basis)
    md = StartUpDG.make_periodic(md, periodicity)
    boundary_faces = StartUpDG.tag_boundary_faces(triangulateIO, dg.basis, md,
                                                  boundary_dict)
    return DGMultiMesh(dg, GeometricTermsType(TriangulateIO(), dg), md, boundary_faces)
end

"""
    DGMultiMesh(dg::DGMulti, cells_per_dimension;
                coordinates_min=(-1.0, -1.0), coordinates_max=(1.0, 1.0),
                is_on_boundary=nothing,
                periodicity=ntuple(_ -> false, NDIMS))

Constructs a Cartesian [`DGMultiMesh`](@ref) with element type `dg.basis.element_type`. The domain is
the tensor product of the intervals `[coordinates_min[i], coordinates_max[i]]`.
- `is_on_boundary` specifies boundary using a `Dict{Symbol, <:Function}`
- `periodicity` is a tuple of `Bool`s specifying periodicity = `true`/`false` in the (x,y,z) direction.
"""
function DGMultiMesh(dg::DGMulti{NDIMS}, cells_per_dimension;
                     coordinates_min = ntuple(_ -> -one(real(dg)), NDIMS),
                     coordinates_max = ntuple(_ -> one(real(dg)), NDIMS),
                     is_on_boundary = nothing,
                     periodicity = ntuple(_ -> false, NDIMS), kwargs...) where {NDIMS}
    vertex_coordinates, EToV = StartUpDG.uniform_mesh(dg.basis.element_type,
                                                      cells_per_dimension...)
    domain_lengths = coordinates_max .- coordinates_min
    for i in 1:NDIMS
        @. vertex_coordinates[i] = 0.5 * (vertex_coordinates[i] + 1) *
                                   domain_lengths[i] + coordinates_min[i]
    end

    md = MeshData(vertex_coordinates, EToV, dg.basis)
    if NDIMS == 1 && first(periodicity) == true
        md = StartUpDG.make_periodic(md)
    end
    if NDIMS > 1
        md = StartUpDG.make_periodic(md, periodicity)
    end
    boundary_faces = StartUpDG.tag_boundary_faces(md, is_on_boundary)
    return DGMultiMesh(dg, GeometricTermsType(Cartesian(), dg), md, boundary_faces)
end

"""
    DGMultiMesh(dg::DGMulti{NDIMS}, cells_per_dimension, mapping;
                is_on_boundary=nothing,
                periodicity=ntuple(_ -> false, NDIMS), kwargs...) where {NDIMS}

Constructs a `Curved()` [`DGMultiMesh`](@ref) with element type `dg.basis.element_type`.
- `mapping` is a function which maps from a reference [-1, 1]^NDIMS domain to a mapped domain,
   e.g., `xy = mapping(x, y)` in 2D.
- `is_on_boundary` specifies boundary using a `Dict{Symbol, <:Function}`
- `periodicity` is a tuple of `Bool`s specifying periodicity = `true`/`false` in the (x,y,z) direction.
"""
function DGMultiMesh(dg::DGMulti{NDIMS}, cells_per_dimension, mapping;
                     is_on_boundary = nothing,
                     periodicity = ntuple(_ -> false, NDIMS), kwargs...) where {NDIMS}
    vertex_coordinates, EToV = StartUpDG.uniform_mesh(dg.basis.element_type,
                                                      cells_per_dimension...)
    md = MeshData(vertex_coordinates, EToV, dg.basis)
    md = NDIMS == 1 ? StartUpDG.make_periodic(md, periodicity...) :
         StartUpDG.make_periodic(md, periodicity)

    @unpack xyz = md
    for i in eachindex(xyz[1])
        new_xyz = mapping(getindex.(xyz, i)...)
        setindex!.(xyz, new_xyz, i)
    end
    md_curved = MeshData(dg.basis, md, xyz...)

    boundary_faces = StartUpDG.tag_boundary_faces(md_curved, is_on_boundary)
    return DGMultiMesh(dg, GeometricTermsType(Curved(), dg), md_curved, boundary_faces)
end

"""
    DGMultiMesh(dg::DGMulti, filename::String)

- `dg::DGMulti` contains information associated with the reference element (e.g., quadrature,
  basis evaluation, differentiation, etc).
- `filename` is a path specifying a `.mesh` file generated by
  [HOHQMesh](https://github.com/trixi-framework/HOHQMesh).
"""
function DGMultiMesh(dg::DGMulti{NDIMS}, filename::String;
                     periodicity = ntuple(_ -> false, NDIMS)) where {NDIMS}
    hohqmesh_data = StartUpDG.read_HOHQMesh(filename)
    md = MeshData(hohqmesh_data, dg.basis)
    md = StartUpDG.make_periodic(md, periodicity)
    boundary_faces = Dict(Pair.(keys(md.mesh_type.boundary_faces),
                                values(md.mesh_type.boundary_faces)))
    return DGMultiMesh(dg, GeometricTermsType(Curved(), dg), md, boundary_faces)
end

# Matrix type for lazy construction of physical differentiation matrices
# Constructs a lazy linear combination of B = ∑_i coeffs[i] * A[i]
struct LazyMatrixLinearCombo{Tcoeffs, N, Tv, TA <: AbstractMatrix{Tv}} <:
       AbstractMatrix{Tv}
    matrices::NTuple{N, TA}
    coeffs::NTuple{N, Tcoeffs}
    function LazyMatrixLinearCombo(matrices, coeffs)
        @assert all(matrix -> size(matrix) == size(first(matrices)), matrices)
        new{typeof(first(coeffs)), length(matrices), eltype(first(matrices)),
            typeof(first(matrices))}(matrices, coeffs)
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
function SimpleKronecker(NDIMS, A, eltype_A = eltype(A))
    @assert size(A, 1) == size(A, 2) # check if square
    tmp_storage = [zeros(eltype_A, ntuple(_ -> size(A, 2), NDIMS)...)
                   for _ in 1:Threads.nthreads()]
    return SimpleKronecker{NDIMS, typeof(A), typeof(tmp_storage)}(A, tmp_storage)
end

# fall back to mul! for a 1D Kronecker product
LinearAlgebra.mul!(b, A_kronecker::SimpleKronecker{1}, x) = mul!(b, A_kronecker.A, x)

# Computes `b = kron(A, A) * x` in an optimized fashion
function LinearAlgebra.mul!(b_in, A_kronecker::SimpleKronecker{2}, x_in)
    @unpack A = A_kronecker
    tmp_storage = A_kronecker.tmp_storage[Threads.threadid()]
    n = size(A, 2)

    # copy `x_in` to `tmp_storage` to avoid mutating the input
    @assert length(tmp_storage) == length(x_in)
    @turbo thread=true for i in eachindex(tmp_storage)
        tmp_storage[i] = x_in[i]
    end
    x = reshape(tmp_storage, n, n)
    # As of Julia 1.9, Base.ReshapedArray does not produce allocations when setting values.
    # Thus, Base.ReshapedArray should be used if you are setting values in the array.
    # `reshape` is fine if you are only accessing values.
    b = Base.ReshapedArray(b_in, (n, n), ())

    @turbo thread=true for j in 1:n, i in 1:n
        tmp = zero(eltype(x))
        for ii in 1:n
            tmp = tmp + A[i, ii] * x[ii, j]
        end
        b[i, j] = tmp
    end

    @turbo thread=true for j in 1:n, i in 1:n
        tmp = zero(eltype(x))
        for jj in 1:n
            tmp = tmp + A[j, jj] * b[i, jj]
        end
        x[i, j] = tmp
    end

    @turbo thread=true for i in eachindex(b_in)
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
    @turbo thread=true for i in eachindex(tmp_storage)
        tmp_storage[i] = x_in[i]
    end
    x = reshape(tmp_storage, n, n, n)
    # As of Julia 1.9, Base.ReshapedArray does not produce allocations when setting values.
    # Thus, Base.ReshapedArray should be used if you are setting values in the array.
    # `reshape` is fine if you are only accessing values.
    b = Base.ReshapedArray(b_in, (n, n, n), ())

    @turbo thread=true for k in 1:n, j in 1:n, i in 1:n
        tmp = zero(eltype(x))
        for ii in 1:n
            tmp = tmp + A[i, ii] * x[ii, j, k]
        end
        b[i, j, k] = tmp
    end

    @turbo thread=true for k in 1:n, j in 1:n, i in 1:n
        tmp = zero(eltype(x))
        for jj in 1:n
            tmp = tmp + A[j, jj] * b[i, jj, k]
        end
        x[i, j, k] = tmp
    end

    @turbo thread=true for k in 1:n, j in 1:n, i in 1:n
        tmp = zero(eltype(x))
        for kk in 1:n
            tmp = tmp + A[k, kk] * x[i, j, kk]
        end
        b[i, j, k] = tmp
    end

    return nothing
end
end # @muladd
