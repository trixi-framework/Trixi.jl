"""
    hadamard_sum_A_transposed!(du, ATr, volume_flux, u, skip_index=(i,j)->false)

Computes the flux difference ∑_j A[i, j] * f(u_i, u_j) and accumulates the result into `du`.
- `du`, `u` are vectors
- `ATr` is the transpose of the flux differencing matrix `A`. The transpose is used for
  faster traversal since matrices are column major in Julia.
"""
@inline function hadamard_sum_A_transposed!(du, ATr, volume_flux, u, skip_index=(i,j)->false)
  rows, cols = axes(ATr)
  for i in cols
    u_i = u[i]
    du_i = du[i]
    for j in rows
      if !skip_index(i,j)
          du_i += ATr[j,i] * volume_flux(u_i, u[j])
      end
    end
    du[i] = du_i
  end
end

# Version of function without `skip_index`, as mentioned in
# https://github.com/trixi-framework/Trixi.jl/pull/695/files#r670403097
@inline function hadamard_sum_A_transposed!(du, ATr, volume_flux::VF, u) where {VF}
  rows, cols = axes(ATr)
  for i in cols
    u_i = u[i]
    du_i = du[i]
    for j in rows
      du_i += ATr[j,i] * volume_flux(u_i, u[j])
    end
    du[i] = du_i
  end
end

# For DGMulti implementations, we construct "physical" differentiation operators by taking linear
# combinations of reference differentiation operators scaled by geometric change of variables terms.
# We use LazyArrays.jl for lazy evaluation of physical differentiation operators, so that we can
# compute linear combinations of differentiation operators on-the-fly in an allocation-free manner.
function build_lazy_physical_derivative(element::Int, orientation::Int,
                                        mesh::VertexMappedMesh{2, Tri}, dg, cache)
  @unpack Qrst_skew_Tr = cache
  @unpack rxJ, sxJ, ryJ, syJ = mesh.md
  QrskewTr, QsskewTr = Qrst_skew_Tr
  if orientation == 1
    return LazyArray(@~ @. 2 * (rxJ[1,element] * QrskewTr + sxJ[1,element] * QsskewTr))
  else
    return LazyArray(@~ @. 2 * (ryJ[1,element] * QrskewTr + syJ[1,element] * QsskewTr))
  end
end

function build_lazy_physical_derivative(element::Int, orientation::Int,
                                        mesh::VertexMappedMesh{3, Tet}, dg, cache)
  @unpack Qrst_skew_Tr = cache
  QrskewTr, QsskewTr, QtskewTr = Qrst_skew_Tr
  @unpack rxJ, sxJ, txJ, ryJ, syJ, tyJ, rzJ, szJ, tzJ = mesh.md
  if orientation == 1
    return LazyArray(@~ @. 2 * (rxJ[1,element]*QrskewTr + sxJ[1,element]*QsskewTr + txJ[1,element]*QtskewTr))
  elseif orientation == 2
    return LazyArray(@~ @. 2 * (ryJ[1,element]*QrskewTr + syJ[1,element]*QsskewTr + tyJ[1,element]*QtskewTr))
  elseif orientation == 3
    return LazyArray(@~ @. 2 * (rzJ[1,element]*QrskewTr + szJ[1,element]*QsskewTr + tzJ[1,element]*QtskewTr))
  end
end

function calc_volume_integral!(du, u, volume_integral,
                               mesh::VertexMappedMesh, equations, dg::DGMultiFluxDiff{<:SBP}, cache)

  rd = dg.basis
  @unpack local_values_threaded = cache

  volume_flux_oriented(i) = @inline (u_ll, u_rr)->volume_integral.volume_flux(u_ll, u_rr, i, equations)

  # Todo: simplices. Dispatch on curved/non-curved mesh types, this code only works for affine meshes (accessing rxJ[1,e],...)
  @threaded for e in eachelement(mesh, dg, cache)
    rhs_local = local_values_threaded[Threads.threadid()]
    fill!(rhs_local, zero(eltype(rhs_local)))
    u_local = view(u, :, e)
    for i in eachdim(mesh)
      Qi_skew_Tr = build_lazy_physical_derivative(e, i, mesh, dg, cache)
      hadamard_sum_A_transposed!(rhs_local, Qi_skew_Tr, volume_flux_oriented(i), u_local)
    end
    view(du, :, e) .+= rhs_local ./ rd.wq
  end
end


function create_cache(mesh::VertexMappedMesh, equations, dg::DGMultiFluxDiff{<:Polynomial}, RealT, uEltype)

  rd = dg.basis
  @unpack md = mesh

  # Todo: simplices. Fix this when StartUpDG v0.11.0 releases: new API `Qrst_hybridized, VhP, Ph = StartUpDG.hybridized_SBP_operators(rd)`
  if ndims(mesh) == 2
    Qr_hybridized, Qs_hybridized, VhP, Ph = StartUpDG.hybridized_SBP_operators(rd)
    Qrst_hybridized = (Qr_hybridized, Qs_hybridized)
  elseif ndims(mesh) == 3
    Qr_hybridized, Qs_hybridized, Qt_hybridized, VhP, Ph = StartUpDG.hybridized_SBP_operators(rd)
    Qrst_hybridized = (Qr_hybridized, Qs_hybridized, Qt_hybridized)
  end
  Qrst_skew_Tr = map(A -> -0.5*(A-A'), Qrst_hybridized)

  nvars = nvariables(equations)

  # storage for all quadrature points (concatenated volume / face quadrature points)
  num_quad_points_total = rd.Nq + rd.Nfq
  entropy_projected_u_values = StructArray{SVector{nvars, uEltype}}(ntuple(_->zeros(num_quad_points_total, md.num_elements), nvars))
  projected_entropy_var_values = similar(entropy_projected_u_values)

  # initialize views into entropy_projected_u_values
  u_values = view(entropy_projected_u_values, 1:rd.Nq, :)
  u_face_values = view(entropy_projected_u_values, rd.Nq+1:num_quad_points_total, :)

  # temp storage for entropy variables at volume quad points
  entropy_var_values = StructArray{SVector{nvars, uEltype}}(ntuple(_->zeros(rd.Nq, md.num_elements), nvars))

  # local storage for interface fluxes, rhs, and source
  flux_face_values = similar(u_face_values)
  rhs_local_threaded = [StructArray{SVector{nvars, uEltype}}(ntuple(_->zeros(num_quad_points_total), nvars)) for _ in 1:Threads.nthreads()]
  local_values_threaded = [StructArray{SVector{nvars, uEltype}}(ntuple(_->zeros(rd.Nq), nvars)) for _ in 1:Threads.nthreads()]

  return (; md, Qrst_skew_Tr, VhP, Ph, invJ = inv.(md.J),
            entropy_var_values, projected_entropy_var_values, entropy_projected_u_values,
            u_values, u_face_values, rhs_local_threaded, flux_face_values, local_values_threaded)
end

function entropy_projection!(cache, u, mesh::VertexMappedMesh,  equations, dg::DGMulti)

  rd = dg.basis
  @unpack Vq = rd
  @unpack VhP, entropy_var_values, u_values, entropy_var_values = cache
  @unpack projected_entropy_var_values, entropy_projected_u_values = cache

  # TODO: simplices. Address hard-coding of `entropy2cons!`
  StructArrays.foreachfield(mul_by!(Vq), u_values, u)
  entropy_var_values .= cons2entropy.(u_values, equations)

  # "VhP" fuses the projection "P" with interpolation to volume and face quadrature "Vh"
  StructArrays.foreachfield(mul_by!(VhP), projected_entropy_var_values, entropy_var_values)
  entropy_projected_u_values .= entropy2cons.(projected_entropy_var_values, equations)
end

function calc_volume_integral!(du, u::StructArray, volume_integral,
                               mesh::VertexMappedMesh, equations, dg::DGMultiFluxDiff{<:Polynomial}, cache)

  rd = dg.basis
  md = mesh.md
  @unpack entropy_projected_u_values, rhs_local_threaded, Ph = cache
  volume_flux_oriented(i) = @inline (u_ll, u_rr)->volume_integral.volume_flux(u_ll, u_rr, i, equations)

  # skips subblock of Qi_skew_Tr which we know is zero by construction
  skip_index(i,j) = i > rd.Nq && j > rd.Nq

  # Todo: simplices. Dispatch on curved/non-curved mesh types, this code only works for affine meshes (accessing rxJ[1,e],...)
  @threaded for e in eachelement(mesh, dg, cache)
    rhs_local = rhs_local_threaded[Threads.threadid()]
    fill!(rhs_local, zero(eltype(rhs_local)))
    u_local = view(entropy_projected_u_values, :, e)
    for i in eachdim(mesh)
      Qi_skew_Tr = build_lazy_physical_derivative(e, i, mesh, dg, cache)
      hadamard_sum_A_transposed!(rhs_local, Qi_skew_Tr, volume_flux_oriented(i), u_local, skip_index)
    end
    StructArrays.foreachfield(mul_by_accum!(Ph), view(du, :, e), rhs_local)
  end
end

# Specializes on Polynomial (e.g., modal) DG methods with a flux differencing volume kernel, e.g.,
# an entropy conservative/stable discretization. For modal DG schemes, an extra `entropy_projection`
# is required (see https://doi.org/10.1016/j.jcp.2018.02.033, Section 4.3).
function rhs!(du, u, t, mesh, equations, initial_condition, boundary_conditions::BC,
              source_terms::Source, dg::DGMultiFluxDiff{Polynomial}, cache) where {Source, BC}

  @trixi_timeit timer() "Reset du/dt" fill!(du, zero(eltype(du)))

  # this function evaluates the solution at volume and face quadrature points (which was previously
  # done in `prolong2interfaces` and `calc_volume_integral`)
  @trixi_timeit timer() "entropy_projection!" entropy_projection!(cache, u, mesh, equations, dg)

  @trixi_timeit timer() "calc_volume_integral!" calc_volume_integral!(du, u, dg.volume_integral,
                                                                      mesh, equations, dg, cache)

  # the following functions are the same as in VolumeIntegralWeakForm, and can be reused from dg.jl
  @trixi_timeit timer() "calc_interface_flux!" calc_interface_flux!(cache, dg.surface_integral, mesh, equations, dg)

  @trixi_timeit timer() "calc_boundary_flux!" calc_boundary_flux!(cache, t, boundary_conditions, mesh, equations, dg)

  @trixi_timeit timer() "calc_surface_integral!" calc_surface_integral!(du, u, dg.surface_integral, mesh, equations, dg, cache)

  @trixi_timeit timer() "invert_jacobian" invert_jacobian!(du, mesh, equations, dg, cache)

  @trixi_timeit timer() "calc_sources!" calc_sources!(du, u, t, source_terms, mesh, equations, dg, cache)

  return nothing
end
