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

function integrate(func::Func, u, mesh::DGMultiMesh,
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
    @batch reduction=(max, linf_divb) for e in eachelement(mesh, dg, cache)
        compute_local_divergence!(local_divB, e, view.(B, :, e), mesh, dg, cache)

        # compute maximum norm
        linf_divB = max(linf_divB, maximum(abs, local_divB))
    end

    return linf_divB
end

# Calculate ∫_e (∂S/∂u ⋅ ∂u/∂t) dΩ_e where the result on element 'e' is kept in reference space
# Note that ∂S/∂u = w(u) with entropy variables w.
# This assumes that both du and u are already interpolated to the quadrature points
function entropy_change_reference_element(du_values_local, u_values_local,
                                          mesh::DGMultiMesh, equations,
                                          dg::DGMulti, cache)
    rd = dg.basis
    @unpack Nq, wq = rd

    # Compute entropy change for this element
    dS_dt_elem = zero(eltype(first(du_values_local)))
    for i in Base.OneTo(Nq) # Loop over quadrature points in the element
        dS_dt_elem += dot(cons2entropy(u_values_local[i], equations),
                          du_values_local[i]) * wq[i]
    end

    return dS_dt_elem
end

# calculate surface integral of func(u, normal_direction, equations) on the reference element.
# For DGMulti, we loop over all faces of the element and integrate using face quadrature weights.
function surface_integral_reference_element(func::Func, u, element,
                                            mesh::DGMultiMesh, equations, dg::DGMulti,
                                            cache, args...) where {Func}
    rd = dg.basis
    @unpack Nfq, wf, Vf = rd
    md = mesh.md
    @unpack nxyzJ = md

    # Interpolate volume solution to face quadrature nodes for this element
    @unpack u_face_local_threaded = cache
    u_face_local = u_face_local_threaded[Threads.threadid()]
    u_elem = view(u, :, element)
    apply_to_each_field(mul_by!(Vf), u_face_local, u_elem)

    surface_integral = zero(eltype(first(u)))
    # Loop over all face nodes for this element
    for i in 1:Nfq
        # Get global face node index (across all elements' face nodes)
        face_node_global = i + (element - 1) * Nfq

        # Get solution at this face node
        u_node = u_face_local[i]

        # Get face normal; nxyzJ stores components as (nxJ, nyJ, nxJ)
        normal_direction = SVector(getindex.(nxyzJ, face_node_global))

        # Multiply with face quadrature weight and accumulate
        surface_integral += wf[i] * func(u_node, normal_direction, equations)
    end

    return surface_integral
end

function create_cache_analysis(analyzer, mesh::DGMultiMesh,
                               equations, dg::DGMulti, cache,
                               RealT, uEltype)
    return (;)
end

SolutionAnalyzer(rd::RefElemData) = rd

nelements(mesh::DGMultiMesh, ::DGMulti, other_args...) = mesh.md.num_elements
function nelementsglobal(mesh::DGMultiMesh, solver::DGMulti, cache)
    if mpi_isparallel()
        error("`nelementsglobal` is not implemented for `DGMultiMesh` when used in parallel with MPI")
    else
        return nelements(mesh, solver)
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
