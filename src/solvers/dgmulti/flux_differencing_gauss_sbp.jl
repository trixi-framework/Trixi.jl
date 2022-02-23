
# ========= GaussSBP approximation types ============
# Note: we define type aliases outside of the @muladd block to avoid Revise breaking when code
# inside the @muladd block is edited. See https://github.com/trixi-framework/Trixi.jl/issues/801
# for more details.

# GaussSBP ApproximationType: e.g., Gauss nodes on quads/hexes
struct GaussSBP end

function tensor_product_quadrature(element_type::Line, r1D, w1D)
  return r1D, w1D
end

function tensor_product_quadrature(element_type::Quad, r1D, w1D)
  sq, rq = vec.(StartUpDG.NodesAndModes.meshgrid(r1D))
  ws, wr = vec.(StartUpDG.NodesAndModes.meshgrid(w1D))
  wq = wr .* ws
  return rq, sq, wq
end

function tensor_product_quadrature(element_type::Hex, r1D, w1D)
  rq, sq, tq = vec.(StartUpDG.NodesAndModes.meshgrid(r1D, r1D, r1D))
  wr, ws, wt = vec.(StartUpDG.NodesAndModes.meshgrid(w1D, w1D, w1D))
  wq = wr .* ws .* wt
  return rq, sq, tq, wq
end

# type parameters for `TensorProductFaceOperator`.
abstract type AbstractGaussOperator end
struct Interpolation <: AbstractGaussOperator end
# - `Projection{ScaleByFaceWeights=Static.False()}` corresponds to the operator `Pf = M \ Vf'`,
#   which is used in `VolumeIntegralFluxDifferencing`.
# - `Projection{ScaleByFaceWeights=Static.True()}` corresponds to the quadrature-based lifting
#   operator `LIFT = M \ (Vf' * diagm(rd.wf))`, which is used in `SurfaceIntegralWeakForm`
struct Projection{ScaleByFaceWeights}  <: AbstractGaussOperator end

#   TensorProductGaussFaceOperator{Tmat, Ti}
#
# Data for performing tensor product interpolation from volume nodes to face nodes.
struct TensorProductGaussFaceOperator{NDIMS, OperatorType <: AbstractGaussOperator,
                                      Tmat, Tweights, Tfweights, Tindices}
  interp_matrix_gauss_to_face_1d::Tmat
  inv_volume_weights_1d::Tweights
  face_weights::Tfweights
  face_indices_tensor_product::Tindices
  nnodes_1d::Int
  nfaces::Int
end

# constructor for a 2D operator
function TensorProductGaussFaceOperator(operator::AbstractGaussOperator,
                                        dg::DGMulti{2, Quad, GaussSBP})
  rd = dg.basis

  rq1D, wq1D = StartUpDG.gauss_quad(0, 0, polydeg(dg))
  interp_matrix_gauss_to_face_1d = polynomial_interpolation_matrix(rq1D, [-1; 1])

  nnodes_1d = length(rq1D)

  # Permutation of indices in a tensor product form
  indices = reshape(1:length(rd.rf), nnodes_1d, rd.Nfaces)
  face_indices_tensor_product = zeros(Int, 2, nnodes_1d, ndims(rd.elementType))
  for i in 1:nnodes_1d # loop over nodes in one face
    face_indices_tensor_product[:, i, 1] .= indices[i, 1:2]
    face_indices_tensor_product[:, i, 2] .= indices[i, 3:4]
  end

  T_op = typeof(operator)
  Tm = typeof(interp_matrix_gauss_to_face_1d)
  Tw = typeof(inv.(wq1D))
  Tf = typeof(rd.wf)
  Ti = typeof(face_indices_tensor_product)
  return TensorProductGaussFaceOperator{2, T_op, Tm, Tw, Tf, Ti}(interp_matrix_gauss_to_face_1d,
                                                                 inv.(wq1D), rd.wf,
                                                                 face_indices_tensor_product,
                                                                 nnodes_1d, rd.Nfaces)
end

# constructor for a 3D operator
function TensorProductGaussFaceOperator(operator::AbstractGaussOperator,
                                        dg::DGMulti{3, Hex, GaussSBP})
  rd = dg.basis

  rq1D, wq1D = StartUpDG.gauss_quad(0, 0, polydeg(dg))
  interp_matrix_gauss_to_face_1d = polynomial_interpolation_matrix(rq1D, [-1; 1])

  nnodes_1d = length(rq1D)

  # Permutation of indices in a tensor product form
  indices = reshape(1:length(rd.rf), nnodes_1d, nnodes_1d, rd.Nfaces)
  face_indices_tensor_product = zeros(Int, 2, nnodes_1d, nnodes_1d, ndims(rd.elementType))
  for j in 1:nnodes_1d, i in 1:nnodes_1d # loop over nodes in one face
    face_indices_tensor_product[:, i, j, 1] .= indices[i, j, 1:2]
    face_indices_tensor_product[:, i, j, 2] .= indices[i, j, 3:4]
    face_indices_tensor_product[:, i, j, 3] .= indices[i, j, 5:6]
  end

  T_op = typeof(operator)
  Tm = typeof(interp_matrix_gauss_to_face_1d)
  Tw = typeof(inv.(wq1D))
  Tf = typeof(rd.wf)
  Ti = typeof(face_indices_tensor_product)
  return TensorProductGaussFaceOperator{3, T_op, Tm, Tw, Tf, Ti}(interp_matrix_gauss_to_face_1d,
                                                                 inv.(wq1D), rd.wf,
                                                                 face_indices_tensor_product,
                                                                 nnodes_1d, rd.Nfaces)
end

# specialize behavior of `mul_by!(A::TensorProductGaussFaceOperator)`
@inline function mul_by!(A::TensorProductGaussFaceOperator)
  return (out, x) -> tensor_product_gauss_face_operator!(out, A, x)
end

@inline function tensor_product_gauss_face_operator!(out::AbstractMatrix,
                                                     A::TensorProductGaussFaceOperator,
                                                     x::AbstractMatrix)
  @threaded for col in Base.OneTo(size(out, 2))
    tensor_product_gauss_face_operator!(view(out, :, col), A, view(x, :, col))
  end
end

# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin

# Interpolates values from volume Gauss nodes to face nodes on one element.
@inline function tensor_product_gauss_face_operator!(out::AbstractVector,
                                                     A::TensorProductGaussFaceOperator{2, Interpolation},
                                                     x::AbstractVector)

  @unpack interp_matrix_gauss_to_face_1d, face_indices_tensor_product = A
  @unpack nnodes_1d, nfaces = A

  fill!(out, zero(eltype(out)))

  # for 2D GaussSBP nodes, the indexing is first in x, then in y
  x = reshape(x, nnodes_1d, nnodes_1d)

  # interpolation in the x-direction
  @turbo for i in Base.OneTo(nnodes_1d) # loop over nodes in a face
    index_left  = face_indices_tensor_product[1, i, 1]
    index_right = face_indices_tensor_product[2, i, 1]
    for jj in Base.OneTo(nnodes_1d)      # loop over "line" of volume nodes
      out[index_left]  = out[index_left]  + interp_matrix_gauss_to_face_1d[1, jj] * x[jj, i]
      out[index_right] = out[index_right] + interp_matrix_gauss_to_face_1d[2, jj] * x[jj, i]
    end
  end

  # interpolation in the y-direction
  @turbo for i in Base.OneTo(nnodes_1d) # loop over nodes in a face
    index_left  = face_indices_tensor_product[1, i, 2]
    index_right = face_indices_tensor_product[2, i, 2]
    for jj in Base.OneTo(nnodes_1d)               # loop over "line" of volume nodes
      out[index_left]  = out[index_left]  + interp_matrix_gauss_to_face_1d[1, jj] * x[i, jj]
      out[index_right] = out[index_right] + interp_matrix_gauss_to_face_1d[2, jj] * x[i, jj]
    end
  end
end

# Interpolates values from volume Gauss nodes to face nodes on one element.
@inline function tensor_product_gauss_face_operator!(out::AbstractVector,
                                                     A::TensorProductGaussFaceOperator{3, Interpolation},
                                                     x::AbstractVector)

  @unpack interp_matrix_gauss_to_face_1d, face_indices_tensor_product = A
  @unpack nnodes_1d, nfaces = A

  fill!(out, zero(eltype(out)))

  # for 3D GaussSBP nodes, the indexing is first in y, then x, then z.
  x = reshape(x, nnodes_1d, nnodes_1d, nnodes_1d)

  # interpolation in the y-direction
  @turbo for j in Base.OneTo(nnodes_1d), i in Base.OneTo(nnodes_1d) # loop over nodes in a face
    index_left  = face_indices_tensor_product[1, i, j, 2]
    index_right = face_indices_tensor_product[2, i, j, 2]
    for jj in Base.OneTo(nnodes_1d) # loop over "line" of volume nodes
      out[index_left]  = out[index_left]  + interp_matrix_gauss_to_face_1d[1, jj] * x[jj, i, j]
      out[index_right] = out[index_right] + interp_matrix_gauss_to_face_1d[2, jj] * x[jj, i, j]
    end
  end

  # interpolation in the x-direction
  @turbo for j in Base.OneTo(nnodes_1d), i in Base.OneTo(nnodes_1d) # loop over nodes in a face
    index_left  = face_indices_tensor_product[1, i, j, 1]
    index_right = face_indices_tensor_product[2, i, j, 1]
    for jj in Base.OneTo(nnodes_1d) # loop over "line" of volume nodes
      out[index_left]  = out[index_left]  + interp_matrix_gauss_to_face_1d[1, jj] * x[i, jj, j]
      out[index_right] = out[index_right] + interp_matrix_gauss_to_face_1d[2, jj] * x[i, jj, j]
    end
  end

  # interpolation in the z-direction
  @turbo for i in Base.OneTo(nnodes_1d), j in Base.OneTo(nnodes_1d) # loop over nodes in a face
    index_left  = face_indices_tensor_product[1, i, j, 3]
    index_right = face_indices_tensor_product[2, i, j, 3]
    for jj in Base.OneTo(nnodes_1d) # loop over "line" of volume nodes
      # The ordering (i,j) -> (j,i) needs to be reversed for this last face.
      # This is due to way we define face nodes for Hex() types in StartUpDG.jl.
      out[index_left]  = out[index_left]  + interp_matrix_gauss_to_face_1d[1, jj] * x[j, i, jj]
      out[index_right] = out[index_right] + interp_matrix_gauss_to_face_1d[2, jj] * x[j, i, jj]
    end
  end
end

# Projects face node values to volume Gauss nodes on one element.
@inline function tensor_product_gauss_face_operator!(out_vec::AbstractVector,
                                                     A::TensorProductGaussFaceOperator{2, Projection{ApplyFaceWeights}},
                                                     x::AbstractVector) where {ApplyFaceWeights}

  @unpack interp_matrix_gauss_to_face_1d, face_indices_tensor_product = A
  @unpack inv_volume_weights_1d, nnodes_1d, nfaces = A

  fill!(out_vec, zero(eltype(out_vec)))

  # for 2D GaussSBP nodes, the indexing is first in x, then y
  out = reshape(out_vec, nnodes_1d, nnodes_1d)

  if ApplyFaceWeights == true
    @turbo for i in eachindex(x)
      x[i] = x[i] * A.face_weights[i]
    end
  end

  # interpolation in the x-direction
  @turbo for i in Base.OneTo(nnodes_1d) # loop over face nodes
    index_left  = face_indices_tensor_product[1, i, 1]
    index_right = face_indices_tensor_product[2, i, 1]
    for jj in Base.OneTo(nnodes_1d) # loop over a line of volume nodes
      out[jj, i] = out[jj, i] + interp_matrix_gauss_to_face_1d[1, jj] * x[index_left]
      out[jj, i] = out[jj, i] + interp_matrix_gauss_to_face_1d[2, jj] * x[index_right]
    end
  end

  # interpolation in the y-direction
  @turbo for i in Base.OneTo(nnodes_1d)
    index_left  = face_indices_tensor_product[1, i, 2]
    index_right = face_indices_tensor_product[2, i, 2]
    # loop over a line of volume nodes
    for jj in Base.OneTo(nnodes_1d)
      out[i, jj] = out[i, jj] + interp_matrix_gauss_to_face_1d[1, jj] * x[index_left]
      out[i, jj] = out[i, jj] + interp_matrix_gauss_to_face_1d[2, jj] * x[index_right]
    end
  end

  # apply inv(M)
  @turbo for j in Base.OneTo(nnodes_1d), i in Base.OneTo(nnodes_1d)
    out[i, j] = out[i, j] * inv_volume_weights_1d[i] * inv_volume_weights_1d[j]
  end
end

# Interpolates values from volume Gauss nodes to face nodes on one element.
@inline function tensor_product_gauss_face_operator!(out_vec::AbstractVector,
                                                     A::TensorProductGaussFaceOperator{3, Projection{ApplyFaceWeights}},
                                                     x::AbstractVector) where {ApplyFaceWeights}

  @unpack interp_matrix_gauss_to_face_1d, face_indices_tensor_product = A
  @unpack inv_volume_weights_1d, nnodes_1d, nfaces = A

  fill!(out_vec, zero(eltype(out_vec)))

  # for 3D GaussSBP nodes, the indexing is first in y, then x, then z.
  out = reshape(out_vec, nnodes_1d, nnodes_1d, nnodes_1d)

  if ApplyFaceWeights == true
    @turbo for i in eachindex(x)
      x[i] = x[i] * A.face_weights[i]
    end
  end

  # interpolation in the y-direction
  @turbo for j in Base.OneTo(nnodes_1d), i in Base.OneTo(nnodes_1d) # loop over nodes in a face
    index_left  = face_indices_tensor_product[1, i, j, 2]
    index_right = face_indices_tensor_product[2, i, j, 2]
    for jj in Base.OneTo(nnodes_1d) # loop over "line" of volume nodes
      out[jj, i, j] = out[jj, i, j] + interp_matrix_gauss_to_face_1d[1, jj] * x[index_left]
      out[jj, i, j] = out[jj, i, j] + interp_matrix_gauss_to_face_1d[2, jj] * x[index_right]
    end
  end

  # interpolation in the x-direction
  @turbo for j in Base.OneTo(nnodes_1d), i in Base.OneTo(nnodes_1d) # loop over nodes in a face
    index_left  = face_indices_tensor_product[1, i, j, 1]
    index_right = face_indices_tensor_product[2, i, j, 1]
    for jj in Base.OneTo(nnodes_1d) # loop over "line" of volume nodes
      out[i, jj, j] = out[i, jj, j] + interp_matrix_gauss_to_face_1d[1, jj] * x[index_left]
      out[i, jj, j] = out[i, jj, j] + interp_matrix_gauss_to_face_1d[2, jj] * x[index_right]
    end
  end

  # interpolation in the z-direction
  @turbo for i in Base.OneTo(nnodes_1d), j in Base.OneTo(nnodes_1d) # loop over nodes in a face
    index_left  = face_indices_tensor_product[1, i, j, 3]
    index_right = face_indices_tensor_product[2, i, j, 3]
    for jj in Base.OneTo(nnodes_1d) # loop over "line" of volume nodes
      # The ordering (i,j) -> (j,i) needs to be reversed for this last face.
      # This is due to way we define face nodes for Hex() types in StartUpDG.jl.
      out[j, i, jj] = out[j, i, jj] + interp_matrix_gauss_to_face_1d[1, jj] * x[index_left]
      out[j, i, jj] = out[j, i, jj] + interp_matrix_gauss_to_face_1d[2, jj] * x[index_right]
    end
  end

  # apply inv(M)
  @turbo for k in Base.OneTo(nnodes_1d), j in Base.OneTo(nnodes_1d), i in Base.OneTo(nnodes_1d)
    out[i, j, k] = out[i, j, k] * inv_volume_weights_1d[i] * inv_volume_weights_1d[j] * inv_volume_weights_1d[k]
  end
end

# Specialized constructor for GaussSBP approximation type on quad elements. Restricting to
# VolumeIntegralFluxDifferencing for now since there isn't a way to exploit this structure
# for VolumeIntegralWeakForm yet.
function DGMulti(element_type::Union{Quad, Hex},
                 approximation_type::GaussSBP,
                 volume_integral::VolumeIntegralFluxDifferencing,
                 surface_integral=SurfaceIntegralWeakForm(surface_flux);
                 polydeg::Integer,
                 surface_flux=flux_central,
                 kwargs...)

  # explicitly create tensor product Gauss quadrature rule with polydeg+1 points
  r1D, w1D = StartUpDG.gauss_quad(0, 0, polydeg)
  gauss_rule_vol = tensor_product_quadrature(element_type, r1D, w1D)
  gauss_rule_face = tensor_product_quadrature(StartUpDG.face_type(element_type), r1D, w1D)

  rd = RefElemData(element_type, Polynomial(), polydeg,
                   quad_rule_vol=gauss_rule_vol,
                   quad_rule_face=gauss_rule_face,
                   kwargs...)

  # Since there is no dedicated GaussSBP approximation type implemented in StartUpDG, we simply
  # initialize `rd = RefElemData(...)` with the appropriate quadrature rules and modify the
  # rd.approximationType manually so we can dispatch on the `GaussSBP` type.
  # This uses the Setfield @set macro, which behaves similarly to `Trixi.remake`.
  rd_gauss = @set rd.approximationType = GaussSBP()

  # We will modify the face interpolation operator of rd_gauss later, but want to do so only after
  # the mesh is initialized, since the face interpolation operator is used for that.
  return DG(rd_gauss, nothing #= mortar =#, surface_integral, volume_integral)
end

# For now, this is mostly the same as `create_cache` for DGMultiFluxDiff{<:Polynomial}.
# In the future, we may modify it so that we can specialize additional parts of GaussSBP() solvers.
function create_cache(mesh::DGMultiMesh, equations,
                      dg::DGMultiFluxDiff{<:GaussSBP, <:Union{Quad, Hex}}, RealT, uEltype)

  rd = dg.basis
  @unpack md = mesh

  cache = invoke(create_cache, Tuple{typeof(mesh), typeof(equations), DGMultiFluxDiff, typeof(RealT), typeof(uEltype)},
                 mesh, equations, dg, RealT, uEltype)

  # for change of basis prior to the volume integral and entropy projection
  r1D, _ = StartUpDG.gauss_lobatto_quad(0, 0, polydeg(dg))
  rq1D, _ = StartUpDG.gauss_quad(0, 0, polydeg(dg))
  interp_matrix_lobatto_to_gauss_1D = polynomial_interpolation_matrix(r1D, rq1D)
  interp_matrix_gauss_to_lobatto_1D = polynomial_interpolation_matrix(rq1D, r1D)
  NDIMS = ndims(rd.elementType)
  interp_matrix_lobatto_to_gauss = SimpleKronecker(NDIMS, interp_matrix_lobatto_to_gauss_1D, uEltype)
  interp_matrix_gauss_to_lobatto = SimpleKronecker(NDIMS, interp_matrix_gauss_to_lobatto_1D, uEltype)
  inv_gauss_weights = inv.(rd.wq)

  # specialized operators to perform tensor product interpolation to faces for Gauss nodes
  interp_matrix_gauss_to_face = TensorProductGaussFaceOperator(Interpolation(), dg)
  Pf = TensorProductGaussFaceOperator(Projection{Static.False()}(), dg)

  # `LIFT` matrix for Gauss nodes - this is equivalent to `Pf` scaled by `diagm(rd.wf)`,
  # where `rd.wf` are Gauss node face quadrature weights.
  gauss_LIFT = TensorProductGaussFaceOperator(Projection{Static.True()}(), dg)

  nvars = nvariables(equations)
  rhs_volume_local_threaded   = [allocate_nested_array(uEltype, nvars, (rd.Nq,), dg)  for _ in 1:Threads.nthreads()]
  gauss_volume_local_threaded = [allocate_nested_array(uEltype, nvars, (rd.Nq,), dg)  for _ in 1:Threads.nthreads()]

  return (; cache..., Pf, gauss_LIFT, inv_gauss_weights,
         rhs_volume_local_threaded, gauss_volume_local_threaded,
         interp_matrix_lobatto_to_gauss, interp_matrix_gauss_to_lobatto,
         interp_matrix_gauss_to_face)
end


# TODO: DGMulti. Address hard-coding of `entropy2cons!` and `cons2entropy!` for this function.
function entropy_projection!(cache, u, mesh::DGMultiMesh, equations, dg::DGMultiFluxDiff{<:GaussSBP})

  rd = dg.basis
  @unpack Vq = rd
  @unpack VhP, entropy_var_values, u_values = cache
  @unpack projected_entropy_var_values, entropy_projected_u_values = cache
  @unpack interp_matrix_lobatto_to_gauss, interp_matrix_gauss_to_face = cache

  @threaded for e in eachelement(mesh, dg, cache)
    apply_to_each_field(mul_by!(interp_matrix_lobatto_to_gauss), view(u_values, :, e), view(u, :, e))
  end

  # transform quadrature values to entropy variables
  cons2entropy!(entropy_var_values, u_values, equations)

  volume_indices = Base.OneTo(rd.Nq)
  face_indices = (rd.Nq + 1):(rd.Nq + rd.Nfq)

  # Interpolate volume Gauss nodes to Gauss face nodes (note the layout of
  # `projected_entropy_var_values = [vol pts; face pts]`).
  entropy_var_face_values = view(projected_entropy_var_values, face_indices, :)
  apply_to_each_field(mul_by!(interp_matrix_gauss_to_face), entropy_var_face_values, entropy_var_values)

  # directly copy over volume values (no entropy projection required)
  entropy_projected_volume_values = view(entropy_projected_u_values, volume_indices, :)
  @threaded for i in eachindex(u_values)
    entropy_projected_volume_values[i] = u_values[i]
  end

  # transform entropy to conservative variables on face values
  entropy_projected_face_values = view(entropy_projected_u_values, face_indices, :)
  entropy2cons!(entropy_projected_face_values, entropy_var_face_values, equations)

  return nothing
end

# Assumes cache.flux_face_values is already computed.
# Enables tensor product evaluation of `LIFT::TensorProductGaussFaceOperator`.
function calc_surface_integral!(du, u, surface_integral::SurfaceIntegralWeakForm,
                                mesh::DGMultiMesh, equations,
                                dg::DGMultiFluxDiff{<:GaussSBP}, cache)
  @unpack gauss_volume_local_threaded, rhs_volume_local_threaded = cache
  @unpack interp_matrix_gauss_to_lobatto, gauss_LIFT = cache

  @threaded for e in eachelement(mesh, dg, cache)

    # applies LIFT matrix, output is stored at Gauss nodes
    gauss_volume_local = gauss_volume_local_threaded[Threads.threadid()]
    apply_to_each_field(mul_by!(gauss_LIFT), gauss_volume_local, view(cache.flux_face_values, :, e))

    # interpolate result back to Lobatto nodes for ease of analysis, visualization
    rhs_volume_local = rhs_volume_local_threaded[Threads.threadid()]
    apply_to_each_field(mul_by!(interp_matrix_gauss_to_lobatto), rhs_volume_local, gauss_volume_local)

    for i in eachindex(rhs_volume_local)
      du[i, e] = du[i, e] + rhs_volume_local[i]
    end
  end
end

function calc_volume_integral!(du, u, mesh::DGMultiMesh,
                               have_nonconservative_terms, equations,
                               volume_integral, dg::DGMultiFluxDiff{<:GaussSBP},
                               cache)

  @unpack entropy_projected_u_values = cache
  @unpack fluxdiff_local_threaded, rhs_local_threaded, rhs_volume_local_threaded = cache

  # After computing the volume integral, we transform back to Lobatto nodes.
  # This allows us to reuse the other DGMulti routines as-is.
  @unpack interp_matrix_gauss_to_lobatto = cache
  @unpack Ph, Pf, inv_gauss_weights = cache

  rd = dg.basis
  volume_indices = Base.OneTo(rd.Nq)
  face_indices = (rd.Nq + 1):(rd.Nq + rd.Nfq)

  @threaded for e in eachelement(mesh, dg, cache)
    fluxdiff_local = fluxdiff_local_threaded[Threads.threadid()]
    fill!(fluxdiff_local, zero(eltype(fluxdiff_local)))
    u_local = view(entropy_projected_u_values, :, e)

    local_flux_differencing!(fluxdiff_local, u_local, e,
                             have_nonconservative_terms, volume_integral,
                             has_sparse_operators(dg),
                             mesh, equations, dg, cache)

    # convert `fluxdiff_local::Vector{<:SVector}` to `rhs_local::StructArray{<:SVector}`
    # for faster performance when using `apply_to_each_field`.
    rhs_local = rhs_local_threaded[Threads.threadid()]
    for i in Base.OneTo(length(fluxdiff_local))
      rhs_local[i] = fluxdiff_local[i]
    end

    # stores rhs contributions only at Gauss volume nodes
    rhs_volume_local = rhs_volume_local_threaded[Threads.threadid()]

    # Here, we exploit that under a Gauss nodal basis the structure of the projection
    # matrix `Ph = [diagm(1 ./ wq), Pf]` such that `Ph * [u; uf] = (u ./ wq) + Pf * uf`.
    local_volume_flux = view(rhs_local, volume_indices)
    local_face_flux = view(rhs_local, face_indices)

    # initialize rhs_volume_local = Pf * local_face_flux
    apply_to_each_field(mul_by!(Pf), rhs_volume_local, local_face_flux)

    # accumulate volume contributions
    for i in eachindex(rhs_volume_local)
      rhs_volume_local[i] = rhs_volume_local[i] + local_volume_flux[i] * inv_gauss_weights[i]
    end

    # transform rhs back to Lobatto nodes
    apply_to_each_field(mul_by!(interp_matrix_gauss_to_lobatto),
                        view(du, :, e), rhs_volume_local)
  end

end


end # @muladd
