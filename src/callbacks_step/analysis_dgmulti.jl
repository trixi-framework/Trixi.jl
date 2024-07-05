# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

function calc_error_norms(func, u, t, analyzer,
                          mesh::DGMultiMesh{NDIMS}, equations, initial_condition,
                          dg::DGMulti{NDIMS}, cache, cache_analysis) where {NDIMS}
    rd = dg.basis
    md = mesh.md
    @unpack u_values = cache

    # interpolate u to quadrature points
    apply_to_each_field(mul_by!(rd.Vq), u_values, u)

    component_l2_errors = zero(eltype(u_values))
    component_linf_errors = zero(eltype(u_values))
    for i in each_quad_node_global(mesh, dg, cache)
        u_exact = initial_condition(SVector(getindex.(md.xyzq, i)), t, equations)
        error_at_node = func(u_values[i], equations) - func(u_exact, equations)
        component_l2_errors += md.wJq[i] * error_at_node .^ 2
        component_linf_errors = max.(component_linf_errors, abs.(error_at_node))
    end
    total_volume = sum(md.wJq)
    return sqrt.(component_l2_errors ./ total_volume), component_linf_errors
end

function integrate(func::Func, u,
                   mesh::DGMultiMesh,
                   equations, dg::DGMulti, cache; normalize = true) where {Func}
    rd = dg.basis
    md = mesh.md
    @unpack u_values = cache

    # interpolate u to quadrature points
    apply_to_each_field(mul_by!(rd.Vq), u_values, u)

    integral = sum(md.wJq .* func.(u_values, equations))
    if normalize == true
        integral /= sum(md.wJq)
    end
    return integral
end

function analyze(::typeof(entropy_timederivative), du, u, t,
                 mesh::DGMultiMesh, equations, dg::DGMulti, cache)
    rd = dg.basis
    md = mesh.md
    @unpack u_values = cache

    # interpolate u, du to quadrature points
    du_values = similar(u_values) # Todo: DGMulti. Can we move this to the analysis cache somehow?
    apply_to_each_field(mul_by!(rd.Vq), du_values, du)
    apply_to_each_field(mul_by!(rd.Vq), u_values, u)

    # compute ∫v(u) * du/dt = ∫dS/dt. We can directly compute v(u) instead of computing the entropy
    # projection here, since the RHS will be projected to polynomials of degree N and testing with
    # the L2 projection of v(u) would be equivalent to testing with v(u) due to the moment-preserving
    # property of the L2 projection.
    dS_dt = zero(eltype(first(du)))
    for i in Base.OneTo(length(md.wJq))
        dS_dt += dot(cons2entropy(u_values[i], equations), du_values[i]) * md.wJq[i]
    end
    return dS_dt
end

# This function is used in `analyze(::Val{:l2_divb},...)` and `analyze(::Val{:linf_divb},...)`
function compute_local_divergence!(local_divergence, element, vector_field,
                                   mesh, dg::DGMulti, cache)
    @unpack md = mesh
    rd = dg.basis
    uEltype = eltype(first(vector_field))

    fill!(local_divergence, zero(uEltype))

    # computes dU_i/dx_i = ∑_j dxhat_j/dx_i * dU_i / dxhat_j
    # dU_i/dx_i is then accumulated into local_divergence.
    # TODO: DGMulti. Extend to curved elements.
    for i in eachdim(mesh)
        for j in eachdim(mesh)
            geometric_scaling = md.rstxyzJ[i, j][1, element]
            jth_ref_derivative_matrix = rd.Drst[j]
            mul!(local_divergence, jth_ref_derivative_matrix, vector_field[i],
                 geometric_scaling, one(uEltype))
        end
    end
end

get_component(u::StructArray, i::Int) = StructArrays.component(u, i)
get_component(u::AbstractArray{<:SVector}, i::Int) = getindex.(u, i)

function analyze(::Val{:l2_divb}, du, u, t,
                 mesh::DGMultiMesh, equations::IdealGlmMhdEquations2D,
                 dg::DGMulti, cache)
    @unpack md = mesh
    rd = dg.basis
    B1 = get_component(u, 6)
    B2 = get_component(u, 7)
    B = (B1, B2)

    uEltype = eltype(B1)
    l2norm_divB = zero(uEltype)
    local_divB = zeros(uEltype, size(B1, 1))
    for e in eachelement(mesh, dg, cache)
        compute_local_divergence!(local_divB, e, view.(B, :, e), mesh, dg, cache)

        # TODO: DGMulti. Extend to curved elements.
        # compute L2 norm squared via J[1, e] * u' * M * u
        local_l2norm_divB = md.J[1, e] * dot(local_divB, rd.M, local_divB)
        l2norm_divB += local_l2norm_divB
    end

    return sqrt(l2norm_divB)
end

function analyze(::Val{:linf_divb}, du, u, t,
                 mesh::DGMultiMesh, equations::IdealGlmMhdEquations2D,
                 dg::DGMulti, cache)
    B1 = get_component(u, 6)
    B2 = get_component(u, 7)
    B = (B1, B2)

    uEltype = eltype(B1)
    linf_divB = zero(uEltype)
    local_divB = zeros(uEltype, size(B1, 1))
    for e in eachelement(mesh, dg, cache)
        compute_local_divergence!(local_divB, e, view.(B, :, e), mesh, dg, cache)

        # compute maximum norm
        linf_divB = max(linf_divB, maximum(abs, local_divB))
    end

    return linf_divB
end

function integrate(func::typeof(enstrophy), u,
                   mesh::DGMultiMesh,
                   equations, equations_parabolic::CompressibleNavierStokesDiffusion3D,
                   dg::DGMulti,
                   cache, cache_parabolic; normalize = true)
    gradients_x, gradients_y, gradients_z = cache_parabolic.gradients

    # allocate local storage for gradients.
    # TODO: can we avoid allocating here?
    local_gradient_quadrature_values = ntuple(_ -> similar(cache_parabolic.local_u_values_threaded),
                                              3)

    integral = zero(eltype(u))
    for e in eachelement(mesh, dg)
        u_quadrature_values = cache_parabolic.local_u_values_threaded[Threads.threadid()]
        gradient_x_quadrature_values = local_gradient_quadrature_values[1][Threads.threadid()]
        gradient_y_quadrature_values = local_gradient_quadrature_values[2][Threads.threadid()]
        gradient_z_quadrature_values = local_gradient_quadrature_values[3][Threads.threadid()]

        # interpolate to quadrature on each element
        apply_to_each_field(mul_by!(dg.basis.Vq), u_quadrature_values, view(u, :, e))
        apply_to_each_field(mul_by!(dg.basis.Vq), gradient_x_quadrature_values,
                            view(gradients_x, :, e))
        apply_to_each_field(mul_by!(dg.basis.Vq), gradient_y_quadrature_values,
                            view(gradients_y, :, e))
        apply_to_each_field(mul_by!(dg.basis.Vq), gradient_z_quadrature_values,
                            view(gradients_z, :, e))

        # integrate over the element
        for i in eachindex(u_quadrature_values)
            gradients_i = SVector(gradient_x_quadrature_values[i],
                                  gradient_y_quadrature_values[i],
                                  gradient_z_quadrature_values[i])
            integral += mesh.md.wJq[i, e] *
                        func(u_quadrature_values[i], gradients_i, equations)
        end
    end
    return integral
end

function create_cache_analysis(analyzer, mesh::DGMultiMesh,
                               equations, dg::DGMulti, cache,
                               RealT, uEltype)
    md = mesh.md
    return (;)
end

SolutionAnalyzer(rd::RefElemData) = rd

nelements(mesh::DGMultiMesh, ::DGMulti, other_args...) = mesh.md.num_elements
function nelementsglobal(mesh::DGMultiMesh, solver::DGMulti, cache)
    if mpi_isparallel()
        error("`nelementsglobal` is not implemented for `DGMultiMesh` when used in parallel with MPI")
    else
        return ndofs(mesh, solver, cache)
    end
end
function ndofsglobal(mesh::DGMultiMesh, solver::DGMulti, cache)
    if mpi_isparallel()
        error("`ndofsglobal` is not implemented for `DGMultiMesh` when used in parallel with MPI")
    else
        return ndofs(mesh, solver, cache)
    end
end
end # @muladd
