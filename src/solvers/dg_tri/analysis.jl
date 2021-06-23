# Todo: make these two functions more efficient. Both currently allocate several fairly large arrays. 
function calc_error_norms(func, u::StructArray, t, analyzer,
                          mesh::AbstractMeshData{Dim}, equations, initial_condition,
                          dg::DG{<:RefElemData{Dim}}, cache, cache_analysis) where {Dim}
    rd = dg.basis
    md = mesh.md
    @unpack u_values = cache

    # interpolate u to quadrature points
    StructArrays.foreachfield(mul_by!(rd.Vq), u_values, u) 

    # convert md.xyz::NTuple{Dim,Matrix} to StructArray for broadcasting
    xyzq = StructArray{SVector{Dim,real(dg)}}(md.xyzq)
    u_exact_values = initial_condition.(xyzq, t, equations) # todo: move to cache

    # `pointwise_error` is a StructArray{SVector{nvariables(equations),real(dg)}}, so to square each entry 
    # we need to apply a double broadcast (x->x.^2). 
    pointwise_error = func.(u_values, equations) - func.(u_exact_values, equations) # todo: for loop to avoid allocations
    component_l2_errors = sum(md.wJq .* (x->x.^2).(pointwise_error)) 
    component_linf_errors = maximum((x->abs.(x)).(pointwise_error))

    return component_l2_errors, component_linf_errors
end

function integrate(func::Func, u,
                   mesh::AbstractMeshData,
                   equations, dg::DG{<:RefElemData}, cache; normalize=true) where {Func}
    rd = dg.basis
    md = mesh.md
    @unpack u_values = cache

    # interpolate u to quadrature points
    StructArrays.foreachfield(mul_by!(rd.Vq), u_values, u) 

    integral = sum(md.wJq .* func.(u_values, equations))
    if normalize == true
        integral /= sum(md.wJq)
    end
    return integral
end

function analyze(::typeof(entropy_timederivative), du, u, t,
                 mesh::AbstractMeshData, equations, dg::DG{<:RefElemData}, cache)

    rd = dg.basis
    md = mesh.md
    @unpack u_values = cache

    # interpolate u, du to quadrature points
    du_values = similar(u_values) # todo: move to cache
    StructArrays.foreachfield(mul_by!(rd.Vq), du_values, du) 
    StructArrays.foreachfield(mul_by!(rd.Vq), u_values, u) 

    # compute ∫v(u) * du/dt = ∫dS/dt. We can directly compute v(u) instead of 
    # computing the entropy projection here, since the RHS will be projected 
    # to polynomials of degree N and testing with the L2 projection of v(u) 
    # would be equivalent to testing with v(u) due to the moment-preserving 
    # property of the L2 projection.
    dS_dt = zero(real(dg))
    for i in Base.OneTo(length(md.wJq)) 
        dS_dt += dot(cons2entropy(u_values[i],equations), du_values[i]) * md.wJq[i]
    end
    return dS_dt
end

function create_cache_analysis(analyzer, mesh::AbstractMeshData,
                              equations, dg::DG{<:RefElemData}, cache,
                              RealT, uEltype)
    return (; )                           
end

## todo: what else is required in the interface for AnalysisCallback?
SolutionAnalyzer(rd::RefElemData) = rd

# can we add mesh as an argument to this?
nelements(solver::DG{<:RefElemData}, cache) = size(cache.u_values,2) 
