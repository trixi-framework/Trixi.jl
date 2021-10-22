
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

  # Since there is no dedicated GSBP approximation type implemented in StartUpDG, we simply
  # initialize `rd = RefElemData(...)` with the appropriate quadrature rules and modify the
  # rd.approximationType manually so we can dispatch on the `GSBP` type.
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

  cache = invoke(create_cache, Tuple{VertexMappedMesh, Any, DGMultiFluxDiff, Any, Any},
                 mesh, equations, dg, RealT, uEltype)

  # for change of basis prior to the volume integral and entropy projection
  @unpack rq, sq = rd
  interp_matrix_lobatto_to_gauss = StartUpDG.vandermonde(rd.elementType, polydeg(dg), rq, sq) / rd.VDM
  interp_matrix_gauss_to_lobatto = inv(interp_matrix_lobatto_to_gauss)
  interp_matrix_gauss_to_face = rd.Vf * interp_matrix_gauss_to_lobatto

  # Projection matrix Pf = inv(M) * Vf' in the Gauss nodal basis.
  # Uses that M is a diagonal matrix with the weights on the diagonal under a Gauss nodal basis.
  Pf = diagm(1 ./ rd.wq) * interp_matrix_gauss_to_face'

  nvars = nvariables(equations)
  rhs_gauss = allocate_nested_array(uEltype, nvars, (rd.Nq, md.num_elements), dg)

  return (; cache..., Pf, rhs_gauss,
         interp_matrix_lobatto_to_gauss, interp_matrix_gauss_to_lobatto, interp_matrix_gauss_to_face)
end

# # This function interpolates to Gauss nodes, then performs the entropy projection step
# function entropy_projection!(cache, u, mesh::VertexMappedMesh, equations, dg::DGMultiFluxDiff{<:GSBP})

#   rd = dg.basis
#   @unpack VhP, entropy_var_values, u_values, u_face_values = cache
#   @unpack projected_entropy_var_values, entropy_projected_u_values = cache

#   # Interpolates nodal values at Lobatto points and stores the values in u_values.
#   # Note that `u_values` is a view into `entropy_projected_u_values`.
#   # We will change the basis back to Lobatto nodes in `calc_volume_integral!`
#   @unpack interpolation_matrix_lobatto_to_gauss = cache
#   apply_to_each_field(mul_by!(interpolation_matrix_lobatto_to_gauss), u_values, u)

#   # Transform `u_values` to entropy variables.
#   @threaded for i in Base.OneTo(length(u_values))
#     entropy_var_values[i] = cons2entropy(u_values[i], equations)
#   end

#   # Interpolate entropy variables to face nodes, store in `u_face_values`.
#   # Note that `u_face_values` is a view into `entropy_projected_u_values`.
#   @unpack interpolation_matrix_gauss_to_face = cache
#   apply_to_each_field(mul_by!(interpolation_matrix_gauss_to_face),
#                       u_face_values, entropy_var_values)

#   # This is an in-place conversion of `u_face_values` from entropy variables back to
#   # conservative variables.
#   @threaded for i in Base.OneTo(length(u_face_values))
#     u_face_values[i] = entropy2cons(u_face_values[i], equations)
#   end
# end

# function calc_volume_integral!(du, u, volume_integral,
#                                mesh::VertexMappedMesh,
#                                have_nonconservative_terms::Val{false}, equations,
#                                dg::DGMultiFluxDiff{<:GSBP}, cache)

#   rd = dg.basis
#   @unpack entropy_projected_u_values, Ph, sparsity_pattern = cache
#   @unpack fluxdiff_local_threaded, rhs_local_threaded, rhs_face_local_threaded, rhs_gauss = cache
#   @unpack volume_flux = volume_integral

#    # After computing the volume integral, we transform back to Lobatto nodes.
#   # This allows us to reuse the other DGMulti routines as-is.
#   @unpack interpolation_matrix_gauss_to_lobatto = cache

#   @threaded for e in eachelement(mesh, dg, cache)
#     fluxdiff_local = fluxdiff_local_threaded[Threads.threadid()]
#     fill!(fluxdiff_local, zero(eltype(fluxdiff_local)))
#     u_local = view(entropy_projected_u_values, :, e)

#     local_flux_differencing!(fluxdiff_local, u_local, e,
#                              have_nonconservative_terms, volume_integral,
#                              has_sparse_operators(dg),
#                              mesh, equations, dg, cache)

#     # convert fluxdiff_local::Vector{<:SVector} to StructArray{<:SVector} for faster
#     # apply_to_each_field performance.
#     rhs_local = rhs_local_threaded[Threads.threadid()]
#     for i in Base.OneTo(length(fluxdiff_local))
#       rhs_local[i] = fluxdiff_local[i]
#     end
#     apply_to_each_field(mul_by_accum!(Ph), view(du, :, e), rhs_local)
#   end

#   apply_to_each_field(mul_by!(interpolation_matrix_gauss_to_lobatto), du, rhs_gauss)

# end
