
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

# Todo: DGMulti. Decide if we should add GaussSBP on triangles.


# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin

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

  # create tensor product Gauss quadrature rule with polydeg+1 points
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
function create_cache(mesh::VertexMappedMesh, equations,
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
  interp_matrix_gauss_to_face = rd.Vf * kron(ntuple(_->interp_matrix_gauss_to_lobatto_1D, NDIMS)...)

  # Projection matrix Pf = inv(M) * Vf' in the Gauss nodal basis.
  # Uses that M is a diagonal matrix with the weights on the diagonal under a Gauss nodal basis.
  inv_gauss_weights = inv.(rd.wq)
  Pf = diagm(inv_gauss_weights) * interp_matrix_gauss_to_face'

  nvars = nvariables(equations)
  rhs_volume_local_threaded = [allocate_nested_array(uEltype, nvars, (rd.Nq,), dg)  for _ in 1:Threads.nthreads()]

  return (; cache..., Pf, inv_gauss_weights, rhs_volume_local_threaded,
         interp_matrix_lobatto_to_gauss, interp_matrix_gauss_to_lobatto,
         interp_matrix_gauss_to_face)
end

# TODO: DGMulti. Address hard-coding of `entropy2cons!` and `cons2entropy!` for this function.
function entropy_projection!(cache, u, mesh::VertexMappedMesh, equations, dg::DGMultiFluxDiff{<:GaussSBP})

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

  # interpolate volume Gauss nodes to face nodes
  # (note the layout of projected_entropy_var_values = [vol pts; face pts]).
  entropy_var_face_values = view(projected_entropy_var_values, face_indices, :)
  # TODO: speed up using sparsity?
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

function calc_volume_integral!(du, u, volume_integral, mesh::VertexMappedMesh,
                               have_nonconservative_terms::Val{false}, equations,
                               dg::DGMultiFluxDiff{<:GaussSBP}, cache)

  @unpack entropy_projected_u_values, sparsity_pattern = cache
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
    # TODO: speed up using sparsity?
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
