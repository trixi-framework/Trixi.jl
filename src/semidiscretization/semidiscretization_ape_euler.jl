struct SemidiscretizationApeEuler{SemiApe, SemiEuler, SourceRegion, Weights, Cache} <: AbstractSemidiscretization
  semi_ape::SemiApe
  semi_euler::SemiEuler
  source_region::SourceRegion # function to determine whether a point x lies in the acoustic source region
  weights::Weights # weighting function for the acoustic source terms, default is 1.0 everywhere
  performance_counter::PerformanceCounter
  cache::Cache

  function SemidiscretizationApeEuler{SemiApe, SemiEuler, SourceRegion, Weights, Cache}(
      semi_ape::SemiApe, semi_euler::SemiEuler, source_region::SourceRegion, weights::Weights,
      cache::Cache) where {SemiApe, SemiEuler, SourceRegion, Weights, Cache}

    mesh_ape, equations_ape, solver_ape, _ = mesh_equations_solver_cache(semi_ape)
    mesh_euler, equations_euler, solver_euler, _ = mesh_equations_solver_cache(semi_euler)

    # Currently both semidiscretizations need to use a shared mesh
    @assert mesh_ape == mesh_euler

    @assert ndims(semi_ape) == ndims(semi_euler)
    @assert polydeg(semi_ape.solver) == polydeg(semi_euler.solver)

    performance_counter = PerformanceCounter()
    new{SemiApe, SemiEuler, typeof(source_region), typeof(weights), Cache}(semi_ape, semi_euler,
                                                                           source_region, weights,
                                                                           performance_counter, cache)
  end
end

# TODO: Default `weights` are based on potentially solver-specific cache structure
function SemidiscretizationApeEuler(semi_ape::SemiApe, semi_euler::SemiEuler;
                                    source_region, weights=x -> one(eltype(semi_ape.cache.elements))) where
    {Mesh, SemiApe<:SemidiscretizationHyperbolic{Mesh, <:AbstractAcousticPerturbationEquations},
     SemiEuler<:SemidiscretizationHyperbolic{Mesh, <:AbstractCompressibleEulerEquations}}
  # Vector for the Euler solution
  u_ode = allocate_coefficients(mesh_equations_solver_cache(semi_euler)...)

  cache = (; u_ode, create_cache(SemidiscretizationApeEuler,
                                 mesh_equations_solver_cache(semi_ape)...)...)

  return SemidiscretizationApeEuler{typeof(semi_ape), typeof(semi_euler),
                                    typeof(source_region), typeof(weights), typeof(cache)}(
    semi_ape, semi_euler, source_region, weights, cache)
end

# TODO: Where should this function live?
function create_cache(::Type{SemidiscretizationApeEuler}, mesh,
                      equations::AcousticPerturbationEquations2D, dg::DG, cache)
  grad_c_mean_sq = zeros(eltype(cache.elements), (ndims(equations), nnodes(dg), nnodes(dg),
                                                  nelements(cache.elements)))
  acoustic_source_terms = zeros(eltype(cache.elements), size(grad_c_mean_sq))

  return (; grad_c_mean_sq, acoustic_source_terms)
end


function Base.show(io::IO, semi::SemidiscretizationApeEuler)
  print(io, "SemidiscretizationApeEuler")
end

function Base.show(io::IO, mime::MIME"text/plain", semi::SemidiscretizationApeEuler)
  @nospecialize semi # reduce precompilation time

  if get(io, :compact, false)
    show(io, semi)
  else
    summary_header(io, "SemidiscretizationApeEuler")
    #summary_line(io, "semidiscretization Acoustic Perturbation", semi.semi_ape |> typeof |> nameof)
    #show(increment_indent(io), mime, semi.semi_ape)
    #summary_line(io, "semidiscretization Euler", semi.semi_euler |> typeof |> nameof)
    #show(increment_indent(io), mime, semi.semi_euler)
    summary_footer(io)
  end
end


# The APE semidiscretization is the main semidiscretization.
@inline function mesh_equations_solver_cache(semi::SemidiscretizationApeEuler)
  return mesh_equations_solver_cache(semi.semi_ape)
end


@inline Base.ndims(semi::SemidiscretizationApeEuler) = ndims(semi.semi_ape)
@inline Base.real(semi::SemidiscretizationApeEuler) = real(semi.semi_ape)


# Computes the coefficients of the initial condition
@inline function compute_coefficients(t, semi::SemidiscretizationApeEuler)
  compute_coefficients(t, semi.semi_ape)
end

@inline function compute_coefficients!(u_ode, t, semi::SemidiscretizationApeEuler)
  compute_coefficients!(u_ode, t, semi.semi_ape)
end


@inline function calc_error_norms(func, u, t, analyzer, semi::SemidiscretizationApeEuler,
                                  cache_analysis)
  calc_error_norms(func, u, t, analyzer, semi.semi_ape, cache_analysis)
end


function rhs!(du_ode, u_ode, semi::SemidiscretizationApeEuler, t)
  @unpack semi_ape, cache = semi

  u_ape = wrap_array(u_ode, semi_ape)
  du_ape = wrap_array(du_ode, semi_ape)

  time_start = time_ns()

  @trixi_timeit timer() "APE rhs!" rhs!(du_ode, u_ode, semi_ape, t)

  @trixi_timeit timer() "add acoustic source terms" begin
    if ndims(semi_ape) == 2
      @views @. du_ape[1, .., :] += cache.acoustic_source_terms[1, .., :]
      @views @. du_ape[2, .., :] += cache.acoustic_source_terms[2, .., :]
    else
      error("ndims $(ndims(semi_ape)) is not supported")
    end
  end

  @trixi_timeit timer() "calc conservative source term" begin
    if ndims(semi_ape) == 2
      calc_conservative_source_term!(du_ape, u_ape, cache.grad_c_mean_sq,
                                     mesh_equations_solver_cache(semi_ape)...)
    else
      error("ndims $(ndims(semi_ape)) is not supported")
    end
  end

  runtime = time_ns() - time_start
  put!(semi.performance_counter, runtime)

  return nothing
end


function calc_conservative_source_term!(du_ape, u_ape, grad_c_mean_sq,
                                        mesh::TreeMesh{2}, equations, dg::DG, cache)
  @threaded for element in eachelement(cache.elements)
    for j in eachnode(dg), i in eachnode(dg)
      u_node = get_node_vars(u_ape, equations, dg, i, j, element)
      v1_prime, v2_prime, p_prime = cons2state(u_node, equations)
      v1_mean, v2_mean, c_mean, rho_mean = cons2mean(u_node, equations)

      du_ape[3, i, j, element] += (rho_mean * v1_prime + v1_mean * p_prime / c_mean^2) * grad_c_mean_sq[1, i, j, element] +
                                  (rho_mean * v2_prime + v2_mean * p_prime / c_mean^2) * grad_c_mean_sq[2, i, j, element]
    end
  end

  return nothing
end