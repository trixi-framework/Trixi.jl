# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin

#   hadamard_sum!(du, A, volume_flux, orientation, u, equations,
#                              sparsity_pattern, skip_index=(i,j)->false)
#
# Computes the flux difference ∑_j A[i, j] * f(u_i, u_j) and accumulates the result into `du`.
# - `du`, `u` are vectors
# - `A` is the skew-symmetric flux differencing matrix.
# - `sparsity_pattern`: either `nothing` or an AbstractSparseMatrix which specifies the sparsity
#   pattern of `A`
@inline function hadamard_sum!(du, A, volume_flux, orientation, u, equations,
                               sparsity_pattern::Nothing, skip_index=(i,j)->false)

  rows, cols = axes(A)
  for i in rows
    u_i = u[i]
    du_i = du[i]
    for j in cols
      # This routine computes only the upper-triangular part of the hadamard sum (A .* F).
      # We avoid computing the lower-triangular part, and instead accumulate those contributions
      # while computing the upper-triangular part (using the fact that A is skew-symmetric and F
      # is symmetric).
      if j > i && !skip_index(i, j)
          AF_ij = A[i,j] * volume_flux(u_i, u[j], orientation, equations)
          du_i = du_i + AF_ij
          du[j] = du[j] - AF_ij
      end
    end
    du[i] = du_i
  end
end

# If `skip_index` isn't specified, set it to `nothing` and dispatch based on `sparsity_pattern`
@inline function hadamard_sum!(du, A, volume_flux, orientation, u, equations, sparsity_pattern)
  return hadamard_sum!(du, A, volume_flux, orientation, u, equations, sparsity_pattern, nothing)
end

@inline function hadamard_sum!(du, A, volume_flux, orientation, u,
                               equations, sparsity_pattern::AbstractSparseMatrix{Bool},
                               skip_index::Union{Function, Nothing})
  n = size(sparsity_pattern, 2)
  rows = rowvals(sparsity_pattern)
  for i in 1:n
    u_i = u[i]
    du_i = du[i]
    for id in nzrange(sparsity_pattern, i)
      j = rows[id]
      # This routine computes only the upper-triangular part of the hadamard sum (A .* F).
      # We avoid computing the lower-triangular part, and instead accumulate those contributions
      # while computing the upper-triangular part (using the fact that A is skew-symmetric and F
      # is symmetric).
      if j > i
        AF_ij = A[i,j] * volume_flux(u_i, u[j], orientation, equations)
        du_i = du_i + AF_ij
        du[j] = du[j] - AF_ij
      end
    end
    du[i] = du_i
  end
end

# For DGMulti implementations, we construct "physical" differentiation operators by taking linear
# combinations of reference differentiation operators scaled by geometric change of variables terms.
# We use a lazy evaluation of physical differentiation operators, so that we can compute linear
# combinations of differentiation operators on-the-fly in an allocation-free manner.
function build_lazy_physical_derivative(element, orientation,
                                        mesh::VertexMappedMesh{2}, dg, cache)
  @unpack Qrst_skew = cache
  @unpack rxJ, sxJ, ryJ, syJ = mesh.md
  if orientation == 1
    return LazyMatrixLinearCombo(Qrst_skew, 2 .* (rxJ[1,element], sxJ[1,element]))
  else # if orientation == 2
    return LazyMatrixLinearCombo(Qrst_skew, 2 .* (ryJ[1,element], syJ[1,element]))
  end
end

function build_lazy_physical_derivative(element, orientation,
                                        mesh::VertexMappedMesh{3}, dg, cache)
  @unpack Qrst_skew = cache
  @unpack rxJ, sxJ, txJ, ryJ, syJ, tyJ, rzJ, szJ, tzJ = mesh.md
  if orientation == 1
    return LazyMatrixLinearCombo(Qrst_skew, 2 .* (rxJ[1,element], sxJ[1,element], txJ[1,element]))
  elseif orientation == 2
    return LazyMatrixLinearCombo(Qrst_skew, 2 .* (ryJ[1,element], syJ[1,element], tyJ[1,element]))
  else # if orientation == 3
    return LazyMatrixLinearCombo(Qrst_skew, 2 .* (rzJ[1,element], szJ[1,element], tzJ[1,element]))
  end
end

function compute_flux_differencing_SBP_matrices(dg::DGMulti)
  rd = dg.basis
  Qrst_hybridized, VhP, Ph = StartUpDG.hybridized_SBP_operators(rd)
  Qrst_skew = map(A -> 0.5 * (A - A'), Qrst_hybridized)
  return Qrst_skew, VhP, Ph
end

function compute_flux_differencing_SBP_matrices(dg::DGMultiFluxDiff{<:SBP})
  rd = dg.basis
  @unpack M, Drst, Pq = rd
  Qrst = map(D -> M * D, Drst)
  Qrst_skew = map(A -> 0.5 * (A - A'), Qrst)
  return Qrst_skew
end

# precompute sparsity pattern for optimized flux differencing routines for tensor product elements
function compute_sparsity_pattern(flux_diff_matrices, dg::DG,
                                  tol = 100 * eps(real(dg))) where {DG <: DGMultiFluxDiff{ApproxType, <:Union{Quad, Hex}}} where {ApproxType}
  sparsity_pattern = sum(map(A->abs.(A), droptol!.(sparse.(flux_diff_matrices), tol))) .!= 0
  return sparsity_pattern
end

compute_sparsity_pattern(flux_diff_matrices, dg::DGMulti) = nothing

# For flux differencing SBP-type approximations, store solutions in Matrix{SVector{nvars}}.
# This results in a slight speedup for `calc_volume_integral!`.
function allocate_nested_array(uEltype, nvars, array_dimensions, dg::DGMultiFluxDiff{<:SBP})
  return zeros(SVector{nvars, uEltype}, array_dimensions...)
end

function create_cache(mesh::VertexMappedMesh, equations, dg::DGMultiFluxDiff{<:SBP}, RealT, uEltype)

  rd = dg.basis
  md = mesh.md

  # for use with flux differencing schemes
  Qrst_skew = compute_flux_differencing_SBP_matrices(dg)
  sparsity_pattern = compute_sparsity_pattern(Qrst_skew, dg)

  nvars = nvariables(equations)

  # Todo: DGMulti. Factor common storage into a struct (MeshDataCache?) for reuse across solvers?
  # storage for volume quadrature values, face quadrature values, flux values
  u_values = allocate_nested_array(uEltype, nvars, size(md.xq), dg)
  u_face_values = allocate_nested_array(uEltype, nvars, size(md.xf), dg)
  flux_face_values = allocate_nested_array(uEltype, nvars, size(md.xf), dg)
  lift_scalings = rd.wf ./ rd.wq[rd.Fmask] # lift scalings for diag-norm SBP operators

  local_values_threaded = [allocate_nested_array(uEltype, nvars, (rd.Nq,), dg) for _ in 1:Threads.nthreads()]

  # Use an array of SVectors (chunks of `nvars` are contiguous in memory) to speed up flux differencing
  fluxdiff_local_threaded = [zeros(SVector{nvars, uEltype}, rd.Nq) for _ in 1:Threads.nthreads()]

  return (; md, Qrst_skew, sparsity_pattern, invJ = inv.(md.J), lift_scalings, inv_wq = inv.(rd.wq),
            u_values, u_face_values, flux_face_values,
            local_values_threaded, fluxdiff_local_threaded)
end

function create_cache(mesh::VertexMappedMesh, equations, dg::DGMultiFluxDiff{<:Polynomial}, RealT, uEltype)

  rd = dg.basis
  @unpack md = mesh

  Qrst_skew, VhP, Ph = compute_flux_differencing_SBP_matrices(dg)
  sparsity_pattern = compute_sparsity_pattern(Qrst_skew, dg)

  nvars = nvariables(equations)

  # temp storage for entropy variables at volume quad points
  entropy_var_values = allocate_nested_array(uEltype, nvars, (rd.Nq, md.num_elements), dg)

  # storage for all quadrature points (concatenated volume / face quadrature points)
  num_quad_points_total = rd.Nq + rd.Nfq
  entropy_projected_u_values = allocate_nested_array(uEltype, nvars, (num_quad_points_total, md.num_elements), dg)
  projected_entropy_var_values = allocate_nested_array(uEltype, nvars, (num_quad_points_total, md.num_elements), dg)

  # For this specific solver, `prolong2interfaces` will not be used anymore.
  # Instead, this step is also performed in `entropy_projection!`. Thus, we set
  # `u_face_values` as a `view` into `entropy_projected_u_values`. We do not do
  # the same for `u_values` since we will use that with LoopVectorization, which
  # cannot handle such views as of v0.12.66, the latest version at the time of writing.
  u_values = allocate_nested_array(uEltype, nvars, size(md.xq), dg)
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

# TODO: DGMulti. Address hard-coding of `entropy2cons!` and `cons2entropy!` for this function.
function entropy_projection!(cache, u, mesh::VertexMappedMesh, equations, dg::DGMulti)

  rd = dg.basis
  @unpack Vq = rd
  @unpack VhP, entropy_var_values, u_values = cache
  @unpack projected_entropy_var_values, entropy_projected_u_values = cache

  apply_to_each_field(mul_by!(Vq), u_values, u)

  # TODO: DGMulti. `@threaded` crashes when using `eachindex(u_values)`.
  # See https://github.com/JuliaSIMD/Polyester.jl/issues/37 for more details.
  @threaded for i in Base.OneTo(length(u_values))
    entropy_var_values[i] = cons2entropy(u_values[i], equations)
  end

  # "VhP" fuses the projection "P" with interpolation to volume and face quadrature "Vh"
  apply_to_each_field(mul_by!(VhP), projected_entropy_var_values, entropy_var_values)

  # TODO: DGMulti. `@threaded` crashes when using `eachindex(projected_entropy_var_values)`.
  # See https://github.com/JuliaSIMD/Polyester.jl/issues/37 for more details.
  @threaded for i in Base.OneTo(length(projected_entropy_var_values))
    entropy_projected_u_values[i] = entropy2cons(projected_entropy_var_values[i], equations)
  end
end

function calc_volume_integral!(du, u, volume_integral,
                               mesh::VertexMappedMesh, equations, dg::DGMultiFluxDiff{<:SBP}, cache)

  rd = dg.basis
  @unpack fluxdiff_local_threaded, sparsity_pattern, inv_wq = cache
  @unpack volume_flux = volume_integral

  # Todo: DGMulti. Dispatch on curved/non-curved mesh types, this code only works for affine meshes (accessing rxJ[1,e],...)
  @threaded for e in eachelement(mesh, dg, cache)
    fluxdiff_local = fluxdiff_local_threaded[Threads.threadid()]
    fill!(fluxdiff_local, zero(eltype(fluxdiff_local)))
    u_local = view(u, :, e)
    for i in eachdim(mesh)
      Qi_skew = build_lazy_physical_derivative(e, i, mesh, dg, cache)

      # use `sparsity_pattern` to dispatch hadamard_sum!. If using Tri or Tet elements, the
      # Qi_skew matrices are dense, and sparsity_pattern = nothing. If using Quad or Hex
      # elements with an SBP approximationType, then sparsity_pattern::AbstractSparseMatrix{Bool}.
      hadamard_sum!(fluxdiff_local, Qi_skew, volume_flux, i,
                    u_local, equations, sparsity_pattern)
    end

    for i in each_quad_node(mesh, dg, cache)
      du[i, e] = du[i, e] + fluxdiff_local[i] * inv_wq[i]
    end
  end
end

function calc_volume_integral!(du, u, volume_integral,
                               mesh::VertexMappedMesh, equations, dg::DGMultiFluxDiff{<:Polynomial}, cache)

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

# Specialize since `u_values` isn't computed for DGMultiFluxDiff{<:SBP} solvers.
function calc_sources!(du, u, t, source_terms,
                       mesh, equations, dg::DGMultiFluxDiff{<:SBP}, cache)

  rd = dg.basis
  md = mesh.md

  @threaded for e in eachelement(mesh, dg, cache)
    for i in each_quad_node(mesh, dg, cache)
      du[i, e] += source_terms(u[i, e], getindex.(md.xyzq, i, e), t, equations)
    end
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
  @trixi_timeit timer() "calc_interface_flux!" calc_interface_flux!(cache, dg.surface_integral,
                                                                    mesh, equations, dg)

  @trixi_timeit timer() "calc_boundary_flux!" calc_boundary_flux!(cache, t, boundary_conditions,
                                                                  mesh, equations, dg)

  @trixi_timeit timer() "calc_surface_integral!" calc_surface_integral!(du, u, dg.surface_integral,
                                                                        mesh, equations, dg, cache)

  @trixi_timeit timer() "invert_jacobian" invert_jacobian!(du, mesh, equations, dg, cache)

  @trixi_timeit timer() "calc_sources!" calc_sources!(du, u, t, source_terms,
                                                      mesh, equations, dg, cache)

  return nothing
end


end # @muladd
