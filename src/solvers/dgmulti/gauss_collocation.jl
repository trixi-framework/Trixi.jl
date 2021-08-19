# the DGMultiFluxDiff{<:GSBP} solver uses create_cache as a "staging" area, where it also
# performs a changes of basis.

function create_cache(mesh::VertexMappedMesh, equations, dg::DGMultiFluxDiff{<:GSBP}, RealT, uEltype)

  rd = dg.basis
  @unpack md = mesh

  Qrst_skew, VhP, Ph = compute_flux_differencing_SBP_matrices(dg)
  sparsity_pattern = compute_sparsity_pattern(Qrst_skew, dg)

  # compute change of basis to Gauss node: uses that rd.Vq (interpolation from nodal to quadrature points)
  # is square and invertible for Gauss collocation.
  rd = @set rd.Vf = rd.Vf / rd.Vq
  rd = @set rd.LIFT = rd.Vq * rd.LIFT
  rd = @set rd.M = diagm(rd.wq)
  # rd = @set rd.

  # Qrst_skew should map from quad nodes to quad nodes, so it doesn't need to be changed.
  # VhP and Ph need to be modified to map to/from the Gauss nodes.
  VhP = VhP / rd.Vq
  Ph = rd.Vq * Ph

  nvars = nvariables(equations)

  # temp storage for entropy variables at volume quad points
  entropy_var_values = allocate_nested_array(uEltype, nvars, (rd.Nq, md.num_elements), dg)

  # storage for all quadrature points (concatenated volume / face quadrature points)
  num_quad_points_total = rd.Nq + rd.Nfq
  entropy_projected_u_values = allocate_nested_array(uEltype, nvars, (num_quad_points_total, md.num_elements), dg)
  projected_entropy_var_values = allocate_nested_array(uEltype, nvars, (num_quad_points_total, md.num_elements), dg)

  # initialize temporary storage as views into entropy_projected_u_values
  u_values = view(entropy_projected_u_values, 1:rd.Nq, :)
  u_face_values = view(entropy_projected_u_values, rd.Nq+1:num_quad_points_total, :)
  flux_face_values = similar(u_face_values)

  # local storage for interface fluxes, rhs, and source
  local_values_threaded = [allocate_nested_array(uEltype, nvars, (rd.Nq,), dg) for _ in 1:Threads.nthreads()]

  # Use an array of SVectors (chunks of `nvars` are contiguous in memory) to speed up flux differencing
  # The result is then transferred to rhs_local_threaded::StructArray{<:SVector} before
  # projecting it and storing it into `du`.
  fluxdiff_local_threaded = [zeros(SVector{nvars, uEltype}, num_quad_points_total) for _ in 1:Threads.nthreads()]
  rhs_local_threaded = [allocate_nested_array(uEltype, nvars, (num_quad_points_total,), dg)  for _ in 1:Threads.nthreads()]

  return (; md, Qrst_skew, sparsity_pattern, VhP, Ph, invJ = inv.(md.J),
            entropy_var_values, projected_entropy_var_values, entropy_projected_u_values,
            u_values, u_face_values,  flux_face_values,
            local_values_threaded, fluxdiff_local_threaded, rhs_local_threaded)
end