# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

# Return the contravariant basis vector corresponding to the Cartesian
# coordinate direction `orientation` in a given `element` of the `mesh`.
# The contravariant basis vectors have entries `dx_i / dxhat_j` where
# j ∈ {1, ..., NDIMS}. Here, `x_i` and `xhat_j` are the ith physical coordinate
# and jth reference coordinate, respectively. These are geometric terms which
# appear when using the chain rule to compute physical derivatives as a linear
# combination of reference derivatives.
@inline function get_contravariant_vector(element, orientation,
                                          mesh::DGMultiMesh{NDIMS}, cache) where {NDIMS}
    # note that rstxyzJ = [rxJ, sxJ, txJ; ryJ syJ tyJ; rzJ szJ tzJ], so that this will return
    # SVector{2}(rxJ[1, element], ryJ[1, element]) in 2D.

    # assumes geometric terms are constant on each element
    dxidxhatj = mesh.md.rstxyzJ
    return SVector{NDIMS}(getindex.(dxidxhatj[:, orientation], 1, element))
end

@inline function get_contravariant_vector(element, orientation,
                                          mesh::DGMultiMesh{NDIMS, NonAffine},
                                          cache) where {NDIMS}
    # note that rstxyzJ = [rxJ, sxJ, txJ; ryJ syJ tyJ; rzJ szJ tzJ]

    # assumes geometric terms vary spatially over each element
    (; dxidxhatj) = cache
    return SVector{NDIMS}(view.(dxidxhatj[:, orientation], :, element))
end

# For Affine meshes, `get_contravariant_vector` returns an SVector of scalars (constant over the
# element). The normal direction is the same for all node pairs.
@inline get_normal_direction(normal_directions::AbstractVector, i, j) = normal_directions

# For NonAffine meshes, `get_contravariant_vector` returns an SVector of per-node arrays.
# We average the normals at nodes i and j for provably entropy-stable de-aliasing of geometric terms.
@inline function get_normal_direction(normal_directions::AbstractVector{<:AbstractVector},
                                      i, j)
    return 0.5f0 * (getindex.(normal_directions, i) + getindex.(normal_directions, j))
end

# use hybridized SBP operators for general flux differencing schemes.
function compute_flux_differencing_SBP_matrices(dg::DGMulti)
    return compute_flux_differencing_SBP_matrices(dg, has_sparse_operators(dg))
end

function compute_flux_differencing_SBP_matrices(dg::DGMulti, sparse_operators)
    rd = dg.basis
    Qrst_hybridized, VhP, Ph = StartUpDG.hybridized_SBP_operators(rd)
    Qrst_skew = map(A -> 0.5 * (A - A'), Qrst_hybridized)
    if sparse_operators == true
        Qrst_skew = map(Qi -> droptol!(sparse(Qi'), 100 * eps(eltype(Qi)))', Qrst_skew)
    end
    return Qrst_skew, VhP, Ph
end

# use traditional multidimensional SBP operators for SBP approximation types.
function compute_flux_differencing_SBP_matrices(dg::DGMultiFluxDiffSBP,
                                                sparse_operators)
    rd = dg.basis
    @unpack M, Drst, Pq = rd
    Qrst = map(D -> M * D, Drst)
    Qrst_skew = map(A -> 0.5 * (A - A'), Qrst)
    if sparse_operators == true
        Qrst_skew = map(Qi -> droptol!(sparse(Qi'), 100 * eps(eltype(Qi)))', Qrst_skew)
    end
    return Qrst_skew
end

# Build element-to-element connectivity from face-to-face connectivity.
# Used for smoothing of shock capturing blending parameters (see `apply_smoothing!`).
#
# Here, `mesh.md.FToF` is a `(num_faces_per_element × num_elements)` array where
# `FToF[f, e]` stores the global face index of the neighbor of local face `f` on
# element `e`. 
#
# Global face indices are laid out as
#
#   global_face_index = (element_index - 1) * num_faces + local_face_index,
# 
# so that the element index can be recovered by integer division:
#
#   element_index = (global_face_index - 1) ÷ num_faces + 1.
# 
# For a non-periodic boundary face, `FToF[f, e]` points back to face `f` of element 
# `e` itself, so boundary elements are listed as their own neighbor.
function build_element_to_element_connectivity(mesh::DGMultiMesh, dg::DGMulti)
    face_to_face_connectivity = mesh.md.FToF
    element_to_element_connectivity = similar(face_to_face_connectivity)
    for e in axes(face_to_face_connectivity, 2)
        for f in axes(face_to_face_connectivity, 1)
            neighbor_face_index = face_to_face_connectivity[f, e]

            # Reverse-engineer element index from face index. Assumes all elements
            # have the same number of faces.
            neighbor_element_index = ((neighbor_face_index - 1) ÷ dg.basis.num_faces) +
                                     1
            element_to_element_connectivity[f, e] = neighbor_element_index
        end
    end
    return element_to_element_connectivity
end

# For flux differencing SBP-type approximations, store solutions in Matrix{SVector{nvars}}.
# This results in a slight speedup for `calc_volume_integral!`.
function allocate_nested_array(uEltype, nvars, array_dimensions, dg::DGMultiFluxDiffSBP)
    return zeros(SVector{nvars, uEltype}, array_dimensions...)
end

function create_cache(mesh::DGMultiMesh, equations, dg::DGMultiFluxDiffSBP,
                      RealT, uEltype)
    rd = dg.basis
    md = mesh.md

    # for use with flux differencing schemes
    Qrst_skew = compute_flux_differencing_SBP_matrices(dg)

    lift_scalings = rd.wf ./ rd.wq[rd.Fmask] # lift scalings for diag-norm SBP operators

    nvars = nvariables(equations)
    # Use an array of SVectors (chunks of `nvars` are contiguous in memory) to speed up flux differencing
    du_local_threaded = [zeros(SVector{nvars, uEltype}, rd.Nq)
                         for _ in 1:Threads.maxthreadid()]

    solution_container = initialize_dgmulti_solution_container(mesh, equations, dg,
                                                               uEltype)

    return (; md, Qrst_skew, dxidxhatj = md.rstxyzJ,
            invJ = inv.(md.J), lift_scalings, inv_wq = inv.(rd.wq),
            solution_container, du_local_threaded)
end

# most general create_cache: works for `DGMultiFluxDiff{<:Polynomial}`
function create_cache(mesh::DGMultiMesh, equations, dg::DGMultiFluxDiff, RealT, uEltype)
    rd = dg.basis
    @unpack md = mesh

    Qrst_skew, VhP, Ph = compute_flux_differencing_SBP_matrices(dg)

    # temp storage for entropy variables at volume quad points
    nvars = nvariables(equations)
    entropy_var_values = allocate_nested_array(uEltype, nvars, (rd.Nq, md.num_elements),
                                               dg)

    # storage for all quadrature points (concatenated volume / face quadrature points)
    num_quad_points_total = rd.Nq + rd.Nfq
    entropy_projected_u_values = allocate_nested_array(uEltype, nvars,
                                                       (num_quad_points_total,
                                                        md.num_elements), dg)
    projected_entropy_var_values = allocate_nested_array(uEltype, nvars,
                                                         (num_quad_points_total,
                                                          md.num_elements), dg)

    # For this specific solver, `prolong2interfaces` will not be used anymore.
    # Instead, this step is also performed in `entropy_projection!`. Thus, we set
    # `u_face_values` as a `view` into `entropy_projected_u_values`. We do not do
    # the same for `u_values` since we will use that with LoopVectorization, which
    # cannot handle such views as of v0.12.66, the latest version at the time of writing.
    u_values = allocate_nested_array(uEltype, nvars, size(md.xq), dg)
    u_face_values = view(entropy_projected_u_values, (rd.Nq + 1):num_quad_points_total,
                         :)
    flux_face_values = similar(u_face_values)

    # local storage for interface fluxes, rhs, and source
    local_values_threaded = [allocate_nested_array(uEltype, nvars, (rd.Nq,), dg)
                             for _ in 1:Threads.maxthreadid()]

    # Use an array of SVectors (chunks of `nvars` are contiguous in memory) to speed up flux differencing
    # The result is then transferred to `rhs_local`, a thread-local element of
    # `rhs_local_threaded::StructArray{<:SVector}` before projecting it and storing it into `du`.
    du_local_threaded = [zeros(SVector{nvars, uEltype}, num_quad_points_total)
                         for _ in 1:Threads.maxthreadid()]
    rhs_local_threaded = [allocate_nested_array(uEltype, nvars,
                                                (num_quad_points_total,), dg)
                          for _ in 1:Threads.maxthreadid()]

    # interpolate geometric terms to both quadrature and face values for curved meshes
    (; Vq, Vf) = dg.basis
    interpolated_geometric_terms = map(x -> [Vq; Vf] * x, mesh.md.rstxyzJ)
    J = Vq * md.J

    solution_container = DGMultiSolutionContainer(u_values, u_face_values,
                                                  flux_face_values,
                                                  local_values_threaded)

    return (; md, Qrst_skew, VhP, Ph,
            invJ = inv.(J), dxidxhatj = interpolated_geometric_terms,
            entropy_var_values, projected_entropy_var_values,
            entropy_projected_u_values,
            solution_container, du_local_threaded, rhs_local_threaded)
end

# TODO: DGMulti. Address hard-coding of `entropy2cons!` and `cons2entropy!` for this function.
function entropy_projection!(cache, u, mesh::DGMultiMesh, equations, dg::DGMulti)
    rd = dg.basis
    @unpack Vq = rd
    @unpack VhP, entropy_var_values = cache
    @unpack projected_entropy_var_values, entropy_projected_u_values = cache
    (; u_values) = cache.solution_container

    apply_to_each_field(mul_by!(Vq), u_values, u)

    cons2entropy!(entropy_var_values, u_values, equations)

    # "VhP" fuses the projection "P" with interpolation to volume and face quadrature "Vh"
    apply_to_each_field(mul_by!(VhP), projected_entropy_var_values, entropy_var_values)

    entropy2cons!(entropy_projected_u_values, projected_entropy_var_values, equations)
    return nothing
end

@inline function cons2entropy!(entropy_var_values::StructArray,
                               u_values::StructArray,
                               equations)
    @threaded for i in eachindex(u_values)
        entropy_var_values[i] = cons2entropy(u_values[i], equations)
    end
end

@inline function entropy2cons!(entropy_projected_u_values::StructArray,
                               projected_entropy_var_values::StructArray,
                               equations)
    @threaded for i in eachindex(projected_entropy_var_values)
        entropy_projected_u_values[i] = entropy2cons(projected_entropy_var_values[i],
                                                     equations)
    end
end

# Trait-like system to dispatch based on whether or not the SBP operators are sparse.
# Designed to be extendable to include specialized `approximation_types` too.
@inline function has_sparse_operators(dg::DGMultiFluxDiff)
    rd = dg.basis
    return has_sparse_operators(rd.element_type, rd.approximation_type)
end

# General fallback for DGMulti solvers:
# Polynomial-based solvers use hybridized SBP operators, which have blocks scaled by outward
# normal components. This implies that operators for different coordinate directions have
# different sparsity patterns. We default to using sum factorization (which is faster when
# operators are sparse) for all `DGMulti` / `StartUpDG.jl` approximation types.
@inline has_sparse_operators(element_type, approx_type) = True()

# For traditional SBP operators on triangles, the operators are fully dense. We avoid using
# sum factorization here, which is slower for fully dense matrices.
@inline function has_sparse_operators(::Union{Line, Tri, Tet},
                                      approx_type::AT) where {AT <: SBP}
    return False()
end

# SBP/GaussSBP operators on quads/hexes use tensor-product operators. Thus, sum factorization is
# more efficient and we use the sparsity structure.
@inline function has_sparse_operators(::Union{Quad, Hex},
                                      approx_type::AT) where {AT <: SBP}
    return True()
end
@inline has_sparse_operators(::Union{Quad, Hex}, approx_type::GaussSBP) = True()

# FD SBP methods have sparse operators
@inline function has_sparse_operators(::Union{Line, Quad, Hex},
                                      approx_type::AbstractDerivativeOperator)
    return True()
end

function calc_volume_integral!(du, u, mesh::DGMultiMesh,
                               have_nonconservative_terms, equations,
                               volume_integral, dg::DGMultiFluxDiff, cache,
                               alpha = true)
    # No interpolation performed for general volume integral.
    # Instead, an element-wise entropy projection (`entropy_projection!`) is performed before, see
    # `rhs!` for `DGMultiFluxDiff`, which populates `entropy_projected_u_values`
    @threaded for element in eachelement(mesh, dg, cache)
        volume_integral_kernel!(du, u, element, mesh,
                                have_nonconservative_terms, equations,
                                volume_integral, dg, cache, alpha)
    end

    return nothing
end

# Computes flux differencing contribution over a single element by looping over node pairs (i, j).
# The physical normal direction for each pair is n_ij = geometric_matrix * ref_entries,
# where ref_entries[d] = Qrst_skew[d][i,j].
# This fuses the NDIMS per-dimension flux
# evaluations of the old dimension-by-dimension loop into a single evaluation per pair.
# Essentially, instead of calculating 
#   volume_flux(u_i, u_j, 1, equations) * Qx[i, j] + volume_flux(u_i, u_j, 2, equations) * Qy[i, j] + ...
# where Qx[i, j] = dr/dx * Qr[i, j] + ds/dx * Qs[i, j], we can expand out and evaluate
#   volume_flux(u_i, u_j, [dr/dx, dr/dy] * Qr[i, j], equations) + 
#   volume_flux(u_i, u_j, [ds/dx, ds/dy] * Qs[i, j], equations)
# which is slightly faster. 
# 
# For dense operators (SBP on Line/Tri/Tet), we do not use this sum factorization trick.
@inline function local_flux_differencing!(du_local, u_local, element_index,
                                          have_nonconservative_terms::False,
                                          volume_flux,
                                          has_sparse_operators::False, mesh,
                                          equations, dg, cache)
    @unpack Qrst_skew = cache
    NDIMS = ndims(mesh)
    row_ids = axes(first(Qrst_skew), 1)
    geometric_matrix = get_contravariant_matrix(element_index, mesh, cache)
    for i in row_ids
        u_i = u_local[i]
        for j in row_ids
            # We use the symmetry of the volume flux and the anti-symmetry
            # of the derivative operator to save half of the volume flux
            # computations.
            if j > i
                u_j = u_local[j]
                ref_entries = SVector(ntuple(d -> Qrst_skew[d][i, j], Val(NDIMS)))
                normal_direction = geometric_matrix * ref_entries
                AF_ij = 2 * volume_flux(u_i, u_j, normal_direction, equations)
                du_local[i] = du_local[i] + AF_ij
                du_local[j] = du_local[j] - AF_ij # Due to skew-symmetry
            end
        end
    end
end

@inline function local_flux_differencing!(du_local, u_local, element_index,
                                          have_nonconservative_terms::True, volume_flux,
                                          has_sparse_operators::False, mesh,
                                          equations, dg, cache)
    @unpack Qrst_skew = cache
    NDIMS = ndims(mesh)
    flux_conservative, flux_nonconservative = volume_flux
    row_ids = axes(first(Qrst_skew), 1)
    geometric_matrix = get_contravariant_matrix(element_index, mesh, cache)
    for i in row_ids
        u_i = u_local[i]
        for j in row_ids
            ref_entries = SVector(ntuple(d -> Qrst_skew[d][i, j], Val(NDIMS)))
            normal_direction = geometric_matrix * ref_entries
            # We use the symmetry of the volume flux and the anti-symmetry
            # of the derivative operator to save half of the volume flux
            # computations.
            if j > i
                u_j = u_local[j]
                AF_ij = 2 * flux_conservative(u_i, u_j, normal_direction, equations)
                du_local[i] = du_local[i] + AF_ij
                du_local[j] = du_local[j] - AF_ij # Due to skew-symmetry
            end
            # Non-conservative terms use the full (non-symmetric) loop.
            # The 0.5f0 factor on the normal direction is necessary for the nonconservative 
            # fluxes based on the interpretation of global SBP operators.  
            # See also `calc_interface_flux!` with `have_nonconservative_terms::True` 
            # in src/solvers/dgsem_tree/dg_1d.jl
            f_nc = flux_nonconservative(u_i, u_local[j], 0.5f0 * normal_direction,
                                        equations)
            du_local[i] = du_local[i] + 2 * f_nc
        end
    end
end

# When the operators are sparse, we use the sum-factorization approach to
# computing flux differencing. Each dimension has its own sparse operator with
# its own sparsity pattern (e.g., tensor-product structure on Quad/Hex elements),
# so we loop per-dimension. For each nonzero entry A[i,j] we evaluate the flux once
# and exploit skew-symmetry to accumulate both the (i,j) and (j,i) contributions.
@inline function local_flux_differencing!(du_local, u_local, element_index,
                                          have_nonconservative_terms::False,
                                          volume_flux,
                                          has_sparse_operators::True, mesh,
                                          equations, dg, cache)
    @unpack Qrst_skew = cache
    for dim in eachdim(mesh)
        normal_directions = get_contravariant_vector(element_index, dim, mesh, cache)
        Q_skew = Qrst_skew[dim]
        A_base, row_ids, rows, vals = sparse_operator_data(Q_skew)
        for i in row_ids
            u_i = u_local[i]
            du_i = du_local[i]
            for id in nzrange(A_base, i)
                j = rows[id]
                # This routine computes only the upper-triangular part of the hadamard sum (A .* F).
                # We avoid computing the lower-triangular part, and instead accumulate those contributions
                # while computing the upper-triangular part (using the fact that A is skew-symmetric and F
                # is symmetric).
                if j > i
                    u_j = u_local[j]
                    A_ij = vals[id]
                    normal_direction_ij = get_normal_direction(normal_directions, i, j)
                    AF_ij = 2 * A_ij *
                            volume_flux(u_i, u_j, normal_direction_ij, equations)
                    du_i = du_i + AF_ij
                    du_local[j] = du_local[j] - AF_ij # Due to skew-symmetry
                end
            end
            du_local[i] = du_i
        end
    end
end

@inline function local_flux_differencing!(du_local, u_local, element_index,
                                          have_nonconservative_terms::True, volume_flux,
                                          has_sparse_operators::True, mesh,
                                          equations, dg, cache)
    @unpack Qrst_skew = cache
    flux_conservative, flux_nonconservative = volume_flux
    for dim in eachdim(mesh)
        normal_directions = get_contravariant_vector(element_index, dim, mesh, cache)
        Q_skew = Qrst_skew[dim]
        A_base, row_ids, rows, vals = sparse_operator_data(Q_skew)
        for i in row_ids
            u_i = u_local[i]
            du_i = du_local[i]
            for id in nzrange(A_base, i)
                j = rows[id]
                A_ij = vals[id]
                u_j = u_local[j]
                normal_direction_ij = get_normal_direction(normal_directions, i, j)
                # Conservative part: exploit skew-symmetry (calculate upper triangular part only).
                if j > i
                    AF_ij = 2 * A_ij *
                            flux_conservative(u_i, u_j, normal_direction_ij, equations)
                    du_i = du_i + AF_ij
                    du_local[j] = du_local[j] - AF_ij # Due to skew-symmetry
                end
                # Non-conservative terms use the full (non-symmetric) loop.
                # The 0.5f0 factor on the normal direction is necessary for the nonconservative 
                # fluxes based on the interpretation of global SBP operators.  
                # See also `calc_interface_flux!` with `have_nonconservative_terms::True` 
                # in src/solvers/dgsem_tree/dg_1d.jl
                f_nc = flux_nonconservative(u_i, u_j, 0.5f0 * normal_direction_ij,
                                            equations)
                du_i = du_i + 2 * A_ij * f_nc
            end
            du_local[i] = du_i
        end
    end
end

# calculates volume integral for <:Polynomial approximation types. We
# do not assume any additional structure (such as collocated volume or
# face nodes, tensor product structure, etc) in `DGMulti`.
@inline function volume_integral_kernel!(du, u, element, mesh::DGMultiMesh,
                                         have_nonconservative_terms, equations,
                                         volume_integral::VolumeIntegralFluxDifferencing,
                                         dg::DGMultiFluxDiff, cache, alpha = true)
    @unpack entropy_projected_u_values, Ph = cache
    @unpack du_local_threaded, rhs_local_threaded = cache

    du_local = du_local_threaded[Threads.threadid()]
    fill!(du_local, zero(eltype(du_local)))
    u_local = view(entropy_projected_u_values, :, element)

    local_flux_differencing!(du_local, u_local, element,
                             have_nonconservative_terms,
                             volume_integral.volume_flux,
                             has_sparse_operators(dg),
                             mesh, equations, dg, cache)

    # convert du_local::Vector{<:SVector} to StructArray{<:SVector} for faster
    # apply_to_each_field performance.
    rhs_local = rhs_local_threaded[Threads.threadid()]
    for i in Base.OneTo(length(du_local))
        rhs_local[i] = du_local[i]
    end
    apply_to_each_field(mul_by_accum!(Ph, alpha), view(du, :, element), rhs_local)

    return nothing
end

@inline function volume_integral_kernel!(du, u, element, mesh::DGMultiMesh,
                                         have_nonconservative_terms, equations,
                                         volume_integral::VolumeIntegralFluxDifferencing,
                                         dg::DGMultiFluxDiffSBP, cache,
                                         alpha = true)
    @unpack du_local_threaded, inv_wq = cache

    du_local = du_local_threaded[Threads.threadid()]
    fill!(du_local, zero(eltype(du_local)))
    u_local = view(u, :, element)

    local_flux_differencing!(du_local, u_local, element,
                             have_nonconservative_terms,
                             volume_integral.volume_flux,
                             has_sparse_operators(dg),
                             mesh, equations, dg, cache)

    for i in each_quad_node(mesh, dg, cache)
        du[i, element] = du[i, element] + alpha * du_local[i] * inv_wq[i]
    end

    return nothing
end

# Specialize since `u_values` isn't computed for DGMultiFluxDiffSBP solvers.
function calc_sources!(du, u, t, source_terms,
                       mesh, equations, dg::DGMultiFluxDiffSBP, cache)
    md = mesh.md

    @threaded for e in eachelement(mesh, dg, cache)
        for i in each_quad_node(mesh, dg, cache)
            du[i, e] += source_terms(u[i, e], SVector(getindex.(md.xyzq, i, e)), t,
                                     equations)
        end
    end
end

# Specializes on Polynomial (e.g., modal) DG methods with a flux differencing volume integral, e.g.,
# an entropy conservative/stable discretization. For modal DG schemes, an extra `entropy_projection!`
# is required (see https://doi.org/10.1016/j.jcp.2018.02.033, Section 4.3).
# Also called by DGMultiFluxDiff{<:GaussSBP} solvers.
function rhs!(du, u, t, mesh, equations, boundary_conditions::BC,
              source_terms::Source, dg::DGMultiFluxDiff, cache) where {Source, BC}
    @trixi_timeit timer() "reset ∂u/∂t" set_zero!(du, dg, cache)

    # this function evaluates the solution at volume and face quadrature points (which was previously
    # done in `prolong2interfaces` and `calc_volume_integral`)
    @trixi_timeit timer() "entropy_projection!" begin
        entropy_projection!(cache, u, mesh, equations, dg)
    end

    @trixi_timeit timer() "volume integral" begin
        calc_volume_integral!(du, u, mesh, have_nonconservative_terms(equations),
                              equations,
                              dg.volume_integral, dg, cache)
    end

    # the following functions are the same as in VolumeIntegralWeakForm, and can be reused from dg.jl
    @trixi_timeit timer() "interface flux" begin
        calc_interface_flux!(cache, dg.surface_integral, mesh,
                             have_nonconservative_terms(equations), equations, dg)
    end

    @trixi_timeit timer() "boundary flux" begin
        calc_boundary_flux!(cache, t, boundary_conditions, mesh,
                            have_nonconservative_terms(equations), equations, dg)
    end

    @trixi_timeit timer() "surface integral" begin
        calc_surface_integral!(du, u, mesh, equations,
                               dg.surface_integral, dg, cache)
    end

    @trixi_timeit timer() "Jacobian" invert_jacobian!(du, mesh, equations, dg, cache)

    @trixi_timeit timer() "source terms" begin
        calc_sources!(du, u, t, source_terms, mesh, equations, dg, cache)
    end

    return nothing
end

# Specializes on SBP (e.g., nodal/collocation) DG methods with a flux differencing volume
# integral, e.g., an entropy conservative/stable discretization. The implementation of `rhs!`
# for such schemes is very similar to the implementation of `rhs!` for standard DG methods,
# but specializes `calc_volume_integral`.
function rhs!(du, u, t, mesh, equations,
              boundary_conditions::BC, source_terms::Source,
              dg::DGMultiFluxDiffSBP, cache) where {BC, Source}
    @trixi_timeit timer() "reset ∂u/∂t" set_zero!(du, dg, cache)

    @trixi_timeit timer() "volume integral" calc_volume_integral!(du, u, mesh,
                                                                  have_nonconservative_terms(equations),
                                                                  equations,
                                                                  dg.volume_integral,
                                                                  dg, cache)

    @trixi_timeit timer() "prolong2interfaces" prolong2interfaces!(cache, u, mesh,
                                                                   equations, dg)

    @trixi_timeit timer() "interface flux" calc_interface_flux!(cache,
                                                                dg.surface_integral,
                                                                mesh,
                                                                have_nonconservative_terms(equations),
                                                                equations, dg)

    @trixi_timeit timer() "boundary flux" calc_boundary_flux!(cache, t,
                                                              boundary_conditions, mesh,
                                                              have_nonconservative_terms(equations),
                                                              equations, dg)

    @trixi_timeit timer() "surface integral" calc_surface_integral!(du, u, mesh,
                                                                    equations,
                                                                    dg.surface_integral,
                                                                    dg, cache)

    @trixi_timeit timer() "Jacobian" invert_jacobian!(du, mesh, equations, dg, cache)

    @trixi_timeit timer() "source terms" calc_sources!(du, u, t, source_terms, mesh,
                                                       equations, dg, cache)

    return nothing
end
end # @muladd
