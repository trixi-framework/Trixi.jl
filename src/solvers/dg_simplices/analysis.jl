# !!! warning "Experimental features"

function calc_error_norms(func, u, t, analyzer,
                          mesh::AbstractMeshData{Dim}, equations, initial_condition,
                          dg::DG{<:RefElemData{Dim}}, cache, cache_analysis) where {Dim}
  rd = dg.basis
  md = mesh.md
  @unpack u_values = cache

  # interpolate u to quadrature points
  StructArrays.foreachfield(mul_by!(rd.Vq), u_values, u)   

  component_l2_errors = zero(eltype(u_values))
  component_linf_errors = zero(eltype(u_values))
  for i in each_quad_node_global(mesh, dg, cache)
    u_exact = initial_condition(getindex.(md.xyzq, i), t, equations)
    error_at_node = func(u_values[i], equations) - func(u_exact, equations)
    component_l2_errors += md.wJq[i] * error_at_node.^2
    component_linf_errors = max.(component_linf_errors, abs.(error_at_node))
  end   
  return sqrt.(component_l2_errors), component_linf_errors
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
  du_values = similar(u_values) # Todo: simplices. Can we move this to the analysis cache somehow?
  StructArrays.foreachfield(mul_by!(rd.Vq), du_values, du) 
  StructArrays.foreachfield(mul_by!(rd.Vq), u_values, u) 

  # compute ∫v(u) * du/dt = ∫dS/dt. We can directly compute v(u) instead of computing the entropy 
  # projection here, since the RHS will be projected to polynomials of degree N and testing with 
  # the L2 projection of v(u) would be equivalent to testing with v(u) due to the moment-preserving 
  # property of the L2 projection. 
  dS_dt = zero(eltype(first(du)))
  for i in Base.OneTo(length(md.wJq)) 
    dS_dt += dot(cons2entropy(u_values[i],equations), du_values[i]) * md.wJq[i]
  end
  return dS_dt
end

function create_cache_analysis(analyzer, mesh::AbstractMeshData,
                               equations, dg::DG{<:RefElemData}, cache,
                               RealT, uEltype)
  md = mesh.md

  return (; )
end

SolutionAnalyzer(rd::RefElemData) = rd

# TODO: simplices. 
# Analysis routines assume a function of this form; should modify to include arguments of the form 
# `nelements(solver, mesh, cache)` instead.
nelements(solver::DG{<:RefElemData}, cache) = size(cache.u_values, 2) 
