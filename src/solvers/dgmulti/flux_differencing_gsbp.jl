
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
  # This uses the Setfield @set macro, which behaves similarly to `Trixi.remake`.
  rd_gauss = @set rd.approximationType = GSBP()

  # We will modify the face interpolation operator of rd_gauss later, but want to do so only after
  # the mesh is initialized, since the face interpolation operator is used for that.
  return DG(rd_gauss, nothing #= mortar =#, surface_integral, volume_integral)
end

# For now, this is mostly the same as `create_cache` for DGMultiFluxDiff{<:Polynomial}.
# In the future, we may modify it so that we can specialize additional parts of GSBP() solvers.
function create_cache(mesh::VertexMappedMesh, equations, dg::DGMultiFluxDiff{<:GSBP}, RealT, uEltype)

  rd = dg.basis
  @unpack md = mesh

  Qrst_skew, VhP, Ph = compute_flux_differencing_SBP_matrices(dg)
  sparsity_pattern = compute_sparsity_pattern(Qrst_skew, dg)

  # for change of basis prior to the volume integral and entropy projection
  r1D, _ = StartUpDG.gauss_quad(0, 0, polydeg(dg))
  rq, sq = vec.(StartUpDG.NodesAndModes.meshgrid(r1D))
  interpolation_matrix_lobatto_to_gauss = StartUpDG.vandermonde(rd.elementType, polydeg(dg), rq, sq) / rd.VDM
  interpolation_matrix_gauss_to_lobatto = inv(interpolation_matrix_lobatto_to_gauss)
  interpolation_matrix_gauss_to_face = rd.Vf * interpolation_matrix_gauss_to_lobatto

  # Projection matrix Pf = inv(M) * Vf' in the Gauss nodal basis.
  # Uses that M is a diagonal matrix with the weights on the diagonal under a Gauss nodal basis.
  Pf = diagm(1 ./ rd.wq) * interpolation_matrix_gauss_to_face'

  nvars = nvariables(equations)

  # temp storage for entropy variables at volume quad points
  entropy_var_values = allocate_nested_array(uEltype, nvars, (rd.Nq, md.num_elements), dg)

  # storage for all quadrature points (concatenated volume / face quadrature points)
  num_quad_points_total = rd.Nq + rd.Nfq
  entropy_projected_u_values = allocate_nested_array(uEltype, nvars, (num_quad_points_total, md.num_elements), dg)
  projected_entropy_var_values = allocate_nested_array(uEltype, nvars, (num_quad_points_total, md.num_elements), dg)

  # For this specific solver, `prolong2interfaces` will not be used anymore.
  # Instead, this step is also performed in `entropy_projection!`. Thus, we set
  # `u_values` and `u_face_values` as a `view` into `entropy_projected_u_values`.
  u_values = view(entropy_projected_u_values, 1:rd.Nq, :)
  u_face_values = view(entropy_projected_u_values, (rd.Nq+1):num_quad_points_total, :)
  flux_face_values = similar(u_face_values)

  # local storage for interface fluxes, rhs, and source
  local_values_threaded = [allocate_nested_array(uEltype, nvars, (rd.Nq,), dg) for _ in 1:Threads.nthreads()]

  # Use an array of SVectors (chunks of `nvars` are contiguous in memory) to speed up flux differencing
  # The result is then transferred to rhs_local_threaded::StructArray{<:SVector} before
  # projecting it and storing it into `du`.
  fluxdiff_local_threaded = [zeros(SVector{nvars, uEltype}, num_quad_points_total) for _ in 1:Threads.nthreads()]
  rhs_local_threaded = [allocate_nested_array(uEltype, nvars, (num_quad_points_total,), dg)  for _ in 1:Threads.nthreads()]
  #rhs_local_threaded = [allocate_nested_array(uEltype, nvars, (rd.Nq,), dg)  for _ in 1:Threads.nthreads()]
  rhs_face_local_threaded = [allocate_nested_array(uEltype, nvars, (rd.Nfq,), dg)  for _ in 1:Threads.nthreads()]

  return (; md, Qrst_skew, sparsity_pattern, VhP, Ph, Pf, invJ = inv.(md.J),
            entropy_var_values, projected_entropy_var_values, entropy_projected_u_values,
            u_values, u_face_values, flux_face_values,
            local_values_threaded, fluxdiff_local_threaded,
            rhs_local_threaded, rhs_face_local_threaded,
            interpolation_matrix_lobatto_to_gauss,
            interpolation_matrix_gauss_to_lobatto,
            interpolation_matrix_gauss_to_face)
end

# This function interpolates to Gauss nodes, then performs the entropy projection step
function entropy_projection!(cache, u, mesh::VertexMappedMesh, equations, dg::DGMultiFluxDiff{<:GSBP})

  rd = dg.basis
  @unpack VhP, entropy_var_values, u_values, u_face_values = cache
  @unpack projected_entropy_var_values, entropy_projected_u_values = cache

  # Interpolates nodal values at Lobatto points and stores the values in u_values.
  # Note that `u_values` is a view into `entropy_projected_u_values`.
  # We will change the basis back to Lobatto nodes in `calc_volume_integral!`
  @unpack interpolation_matrix_lobatto_to_gauss = cache
  apply_to_each_field(mul_by!(interpolation_matrix_lobatto_to_gauss), u_values, u)

  # Transform `u_values` to entropy variables.
  @threaded for i in Base.OneTo(length(u_values))
    entropy_var_values[i] = cons2entropy(u_values[i], equations)
  end

  # Interpolate entropy variables to face nodes, store in `u_face_values`.
  # Note that `u_face_values` is a view into `entropy_projected_u_values`.
  @unpack interpolation_matrix_gauss_to_face = cache
  apply_to_each_field(mul_by!(interpolation_matrix_gauss_to_face),
                      u_face_values, entropy_var_values)

  # This is an in-place conversion of `u_face_values` from entropy variables back to
  # conservative variables.
  @threaded for i in Base.OneTo(length(u_face_values))
    u_face_values[i] = entropy2cons(u_face_values[i], equations)
  end
end

function calc_volume_integral!(du, u, volume_integral,
                               mesh::VertexMappedMesh, equations, dg::DGMultiFluxDiff{<:GSBP}, cache)
  rd = dg.basis
  @unpack entropy_projected_u_values, Ph, sparsity_pattern = cache
  @unpack fluxdiff_local_threaded, rhs_local_threaded = cache
  @unpack volume_flux = volume_integral

  # skips subblock of Qi_skew which we know is zero by construction
  skip_index(i,j) = i > rd.Nq && j > rd.Nq

  # Todo: DGMulti. Dispatch on curved/non-curved mesh types, this code only works for affine meshes (accessing rxJ[1,e],...)
  @threaded for e in eachelement(mesh, dg, cache)
    fluxdiff_local = fluxdiff_local_threaded[Threads.threadid()]
    fill!(fluxdiff_local, zero(eltype(fluxdiff_local)))
    u_local = view(entropy_projected_u_values, :, e)
    for i in eachdim(mesh)
      Qi_skew = build_lazy_physical_derivative(e, i, mesh, dg, cache)
      hadamard_sum!(fluxdiff_local, Qi_skew, volume_flux, i,
                    u_local, equations, sparsity_pattern, skip_index)
    end

    # convert fluxdiff_local::Vector{<:SVector} to StructArray{<:SVector} for faster
    # apply_to_each_field performance.
    rhs_local = rhs_local_threaded[Threads.threadid()]
    for i in Base.OneTo(length(fluxdiff_local))
      rhs_local[i] = fluxdiff_local[i]
    end
    apply_to_each_field(mul_by_accum!(Ph), view(du, :, e), rhs_local)
  end
end

# function calc_volume_integral!(du, u, volume_integral,
#                                mesh::VertexMappedMesh, equations, dg::DGMultiFluxDiff{<:GSBP}, cache)

#   rd = dg.basis
#   @unpack entropy_projected_u_values, Pf, sparsity_pattern = cache
#   @unpack fluxdiff_local_threaded, rhs_local_threaded, rhs_face_local_threaded = cache
#   @unpack volume_flux = volume_integral

#   # After computing the volume integral, we transform back to Lobatto nodes.
#   # This allows us to reuse the other DGMulti routines as-is.
#   @unpack interpolation_matrix_gauss_to_lobatto = cache

#   # Todo: DGMulti. Dispatch on curved/non-curved mesh types, this code only works for affine meshes (accessing rxJ[1,e],...)
#   @threaded for e in eachelement(mesh, dg, cache)
#     fluxdiff_local = fluxdiff_local_threaded[Threads.threadid()]
#     fill!(fluxdiff_local, zero(eltype(fluxdiff_local)))
#     u_local = view(entropy_projected_u_values, :, e)
#     for i in eachdim(mesh)
#       Qi_skew = build_lazy_physical_derivative(e, i, mesh, dg, cache)
#       hadamard_sum!(fluxdiff_local, Qi_skew, volume_flux, i,
#                     u_local, equations, sparsity_pattern)
#     end

#     # Specializes du = Ph * rhs_local in two steps:
#     # 1. rhs_local += fluxdiff_local[1:Nq, :]
#     # 2. rhs_local += invM * Vf^T * fluxdiff_local[Nq+1:end, :] = Pf * fluxdiff_local[Nq+1:end]
#     # 3. Changes basis for `rhs_local` from Gauss back to Lobatto nodes and accumulate into `du`.

#     # accumulate contributions from volume nodes
#     rhs_local = rhs_local_threaded[Threads.threadid()]
#     for i in Base.OneTo(rd.Nq)
#       rhs_local[i, e] = fluxdiff_local[i]
#     end

#     # convert fluxdiff_local::Vector{<:SVector} to StructArray{<:SVector} for faster
#     # apply_to_each_field performance.
#     rhs_face_local = rhs_face_local_threaded[Threads.threadid()]
#     for i in Base.OneTo(rd.Nfq)
#       rhs_face_local[i] = fluxdiff_local[i + rd.Nq]
#     end
#     apply_to_each_field(mul_by_accum!(Pf), rhs_local, rhs_face_local)

#     # change of basis from Gauss back to Lobatto nodes.
#     apply_to_each_field(mul_by!(interpolation_matrix_gauss_to_lobatto), view(du, :, e), rhs_local)
#   end

# end
