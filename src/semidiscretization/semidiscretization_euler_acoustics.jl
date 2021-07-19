struct SemidiscretizationEulerAcoustics{SemiAcoustics, SemiEuler, SourceRegion, Weights, Cache} <: AbstractSemidiscretization
  semi_acoustics::SemiAcoustics
  semi_euler::SemiEuler
  source_region::SourceRegion # function to determine whether a point x lies in the acoustic source region
  weights::Weights # weighting function for the acoustic source terms on a point x, default is 1.0
  performance_counter::PerformanceCounter
  cache::Cache

  function SemidiscretizationEulerAcoustics{SemiAcoustics, SemiEuler, SourceRegion, Weights, Cache}(
    semi_acoustics, semi_euler, source_region, weights, cache) where {SemiAcoustics, SemiEuler,
                                                                      SourceRegion, Weights, Cache}

    # Currently both semidiscretizations need to use a shared mesh
    @assert semi_acoustics.mesh == semi_euler.mesh

    @assert ndims(semi_acoustics) == ndims(semi_euler)
    @assert polydeg(semi_acoustics.solver) == polydeg(semi_euler.solver)

    performance_counter = PerformanceCounter()
    new(semi_acoustics, semi_euler, source_region, weights, performance_counter, cache)
  end
end

# TODO: Default `weights` are based on potentially solver-specific cache structure
function SemidiscretizationEulerAcoustics(semi_acoustics::SemiAcoustics, semi_euler::SemiEuler;
                                          source_region,
                                          weights=x -> one(eltype(semi_acoustics.cache.elements))) where
    {Mesh, SemiAcoustics<:SemidiscretizationHyperbolic{Mesh, <:AbstractAcousticPerturbationEquations},
     SemiEuler<:SemidiscretizationHyperbolic{Mesh, <:AbstractCompressibleEulerEquations}}

  cache = create_cache(SemidiscretizationEulerAcoustics,
                       mesh_equations_solver_cache(semi_acoustics)...)

  return SemidiscretizationEulerAcoustics{typeof(semi_acoustics), typeof(semi_euler),
                                          typeof(source_region), typeof(weights), typeof(cache)}(
    semi_acoustics, semi_euler, source_region, weights, cache)
end

# TODO: Where should this function live?
function create_cache(::Type{SemidiscretizationEulerAcoustics}, mesh,
                      equations::AcousticPerturbationEquations2D, dg::DGSEM, cache)
  grad_c_mean_sq = zeros(eltype(cache.elements), (ndims(equations), nnodes(dg), nnodes(dg),
                                                  nelements(cache.elements)))
  acoustic_source_terms = zeros(eltype(cache.elements), size(grad_c_mean_sq))

  return (; grad_c_mean_sq, acoustic_source_terms)
end


function Base.show(io::IO, semi::SemidiscretizationEulerAcoustics)
  @nospecialize semi # reduce precompilation time

  print(io, "SemidiscretizationApeEuler(")
  print(io,       semi.semi_acoustics)
  print(io, ", ", semi.semi_euler)
  print(io, ", ", semi.source_region)
  print(io, ", ", semi.weights)
  print(io, ", cache(")
  for (idx, key) in enumerate(keys(semi.cache))
    idx > 1 && print(io, " ")
    print(io, key)
  end
  print(io, "))")
end

function Base.show(io::IO, mime::MIME"text/plain", semi::SemidiscretizationEulerAcoustics)
  @nospecialize semi # reduce precompilation time

  if get(io, :compact, false)
    show(io, semi)
  else
    summary_header(io, "SemidiscretizationApeEuler")
    summary_line(io, "semidiscretization Euler", semi.semi_euler |> typeof |> nameof)
    show(increment_indent(io), mime, semi.semi_euler)
    summary_line(io, "semidiscretization acoustics", semi.semi_acoustics |> typeof |> nameof)
    show(increment_indent(io), mime, semi.semi_acoustics)
    summary_line(io, "acoustic source region", semi.source_region |> typeof |> nameof)
    summary_line(io, "acoustic source weights", semi.weights |> typeof |> nameof)
    summary_footer(io)
  end
end


# The acoustics semidiscretization is the main semidiscretization.
@inline function mesh_equations_solver_cache(semi::SemidiscretizationEulerAcoustics)
  return mesh_equations_solver_cache(semi.semi_acoustics)
end


@inline Base.ndims(semi::SemidiscretizationEulerAcoustics) = ndims(semi.semi_acoustics)
@inline Base.real(semi::SemidiscretizationEulerAcoustics) = real(semi.semi_acoustics)


# Computes the coefficients of the initial condition
@inline function compute_coefficients(t, semi::SemidiscretizationEulerAcoustics)
  compute_coefficients(t, semi.semi_acoustics)
end

@inline function compute_coefficients!(u_ode, t, semi::SemidiscretizationEulerAcoustics)
  compute_coefficients!(u_ode, t, semi.semi_acoustics)
end


@inline function calc_error_norms(func, u, t, analyzer, semi::SemidiscretizationEulerAcoustics,
                                  cache_analysis)
  calc_error_norms(func, u, t, analyzer, semi.semi_acoustics, cache_analysis)
end


function rhs!(du_ode, u_ode, semi::SemidiscretizationEulerAcoustics, t)
  @unpack semi_acoustics, cache = semi

  u_acoustics = wrap_array(u_ode, semi_acoustics)
  du_acoustics = wrap_array(du_ode, semi_acoustics)

  time_start = time_ns()

  @trixi_timeit timer() "acoustics rhs!" rhs!(du_ode, u_ode, semi_acoustics, t)

  @trixi_timeit timer() "add acoustic source terms" begin
    if ndims(semi_acoustics) == 2
      @views @. du_acoustics[1, .., :] += cache.acoustic_source_terms[1, .., :]
      @views @. du_acoustics[2, .., :] += cache.acoustic_source_terms[2, .., :]
    else
      error("ndims $(ndims(semi_acoustics)) is not supported")
    end
  end

  @trixi_timeit timer() "calc conservation source term" begin
    if ndims(semi_acoustics) == 2
      calc_conservation_source_term!(du_acoustics, u_acoustics, cache.grad_c_mean_sq,
                                     mesh_equations_solver_cache(semi_acoustics)...)
    else
      error("ndims $(ndims(semi_acoustics)) is not supported")
    end
  end

  runtime = time_ns() - time_start
  put!(semi.performance_counter, runtime)

  return nothing
end


function calc_conservation_source_term!(du_acoustics, u_acoustics, grad_c_mean_sq,
                                        mesh::TreeMesh{2}, equations::AcousticPerturbationEquations2D,
                                        dg::DGSEM, cache)
  @threaded for element in eachelement(cache.elements)
    for j in eachnode(dg), i in eachnode(dg)
      u_node = get_node_vars(u_acoustics, equations, dg, i, j, element)
      v1_prime, v2_prime, p_prime = cons2state(u_node, equations)
      v1_mean, v2_mean, c_mean, rho_mean = cons2mean(u_node, equations)

      du_acoustics[3, i, j, element] += (rho_mean * v1_prime + v1_mean * p_prime / c_mean^2) * grad_c_mean_sq[1, i, j, element] +
                                        (rho_mean * v2_prime + v2_mean * p_prime / c_mean^2) * grad_c_mean_sq[2, i, j, element]
    end
  end

  return nothing
end