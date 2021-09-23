# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin


# TODO: Taal refactor
# - analysis_interval part as PeriodicCallback called after a certain amount of simulation time
"""
    AnalysisCallback(semi; interval=0,
                           save_analysis=false,
                           output_directory="out",
                           analysis_filename="analysis.dat",
                           extra_analysis_errors=Symbol[],
                           extra_analysis_integrals=())

Analyze a numerical solution every `interval` time steps and print the
results to the screen. If `save_analysis`, the results are also saved in
`joinpath(output_directory, analysis_filename)`.

Additional errors can be computed, e.g. by passing `extra_analysis_errors = [:primitive]`.

Further scalar functions `func` in `extra_analysis_integrals` are applied to the numerical
solution and integrated over the computational domain.
See `Trixi.analyze`, `Trixi.pretty_form_utf`, `Trixi.pretty_form_ascii` for further
information on how to create custom analysis quantities.
"""
mutable struct AnalysisCallback{Analyzer, AnalysisIntegrals, InitialStateIntegrals, Cache}
  start_time::Float64
  interval::Int
  save_analysis::Bool
  output_directory::String
  analysis_filename::String
  analyzer::Analyzer
  analysis_errors::Vector{Symbol}
  analysis_integrals::AnalysisIntegrals
  initial_state_integrals::InitialStateIntegrals
  cache::Cache
end


# TODO: Taal bikeshedding, implement a method with less information and the signature
# function Base.show(io::IO, analysis_callback::AnalysisCallback)
# end
function Base.show(io::IO, ::MIME"text/plain", cb::DiscreteCallback{<:Any, <:AnalysisCallback})
  @nospecialize cb # reduce precompilation time

  if get(io, :compact, false)
    show(io, cb)
  else
    analysis_callback = cb.affect!

    setup = Pair{String,Any}[
             "interval" => analysis_callback.interval,
             "analyzer" => analysis_callback.analyzer,
            ]
    for (idx, error) in enumerate(analysis_callback.analysis_errors)
      push!(setup, "│ error " * string(idx) => error)
    end
    for (idx, integral) in enumerate(analysis_callback.analysis_integrals)
      push!(setup, "│ integral " * string(idx) => integral)
    end
    push!(setup, "save analysis to file" => analysis_callback.save_analysis ? "yes" : "no")
    if analysis_callback.save_analysis
      push!(setup, "│ filename" => analysis_callback.analysis_filename)
      push!(setup, "│ output directory" => abspath(normpath(analysis_callback.output_directory)))
    end
    summary_box(io, "AnalysisCallback", setup)
  end
end


function AnalysisCallback(semi::AbstractSemidiscretization; kwargs...)
  mesh, equations, solver, cache = mesh_equations_solver_cache(semi)
  AnalysisCallback(mesh, equations, solver, cache; kwargs...)
end

function AnalysisCallback(mesh, equations::AbstractEquations, solver, cache;
                          interval=0,
                          save_analysis=false,
                          output_directory="out",
                          analysis_filename="analysis.dat",
                          extra_analysis_errors=Symbol[],
                          analysis_errors=union(default_analysis_errors(equations), extra_analysis_errors),
                          extra_analysis_integrals=(),
                          analysis_integrals=union(default_analysis_integrals(equations), extra_analysis_integrals),
                          RealT=real(solver),
                          uEltype=eltype(cache.elements),
                          kwargs...)
  # Decide when the callback is activated.
  # With error-based step size control, some steps can be rejected. Thus,
  #   `integrator.iter >= integrator.destats.naccept`
  #    (total #steps)       (#accepted steps)
  # We need to check the number of accepted steps since callbacks are not
  # activated after a rejected step.
  condition = (u, t, integrator) -> interval > 0 && ( (integrator.destats.naccept % interval == 0 &&
                                                       !(integrator.destats.naccept == 0 && integrator.iter > 0)) ||
                                                     isfinished(integrator))

  analyzer = SolutionAnalyzer(solver; kwargs...)
  cache_analysis = create_cache_analysis(analyzer, mesh, equations, solver, cache, RealT, uEltype)

  analysis_callback = AnalysisCallback(0.0, interval, save_analysis, output_directory, analysis_filename,
                                       analyzer,
                                       analysis_errors, Tuple(analysis_integrals),
                                       SVector(ntuple(_ -> zero(uEltype), Val(nvariables(equations)))),
                                       cache_analysis)

  DiscreteCallback(condition, analysis_callback,
                   save_positions=(false,false),
                   initialize=initialize!)
end


function initialize!(cb::DiscreteCallback{Condition,Affect!}, u_ode, t, integrator) where {Condition, Affect!<:AnalysisCallback}
  semi = integrator.p
  initial_state_integrals = integrate(u_ode, semi)
  _, equations, _, _ = mesh_equations_solver_cache(semi)

  analysis_callback = cb.affect!
  analysis_callback.initial_state_integrals = initial_state_integrals
  @unpack save_analysis, output_directory, analysis_filename, analysis_errors, analysis_integrals = analysis_callback

  if save_analysis && mpi_isroot()
    mkpath(output_directory)

    # write header of output file
    open(joinpath(output_directory, analysis_filename), "w") do io
      @printf(io, "#%-8s", "timestep")
      @printf(io, "  %-14s", "time")
      @printf(io, "  %-14s", "dt")
      if :l2_error in analysis_errors
        for v in varnames(cons2cons, equations)
          @printf(io, "   %-14s", "l2_" * v)
        end
      end
      if :linf_error in analysis_errors
        for v in varnames(cons2cons, equations)
          @printf(io, "   %-14s", "linf_" * v)
        end
      end
      if :conservation_error in analysis_errors
        for v in varnames(cons2cons, equations)
          @printf(io, "   %-14s", "cons_" * v)
        end
      end
      if :residual in analysis_errors
        for v in varnames(cons2cons, equations)
          @printf(io, "   %-14s", "res_" * v)
        end
      end
      if :l2_error_primitive in analysis_errors
        for v in varnames(cons2prim, equations)
          @printf(io, "   %-14s", "l2_" * v)
        end
      end
      if :linf_error_primitive in analysis_errors
        for v in varnames(cons2prim, equations)
          @printf(io, "   %-14s", "linf_" * v)
        end
      end

      for quantity in analysis_integrals
        @printf(io, "   %-14s", pretty_form_ascii(quantity))
      end

      println(io)
    end

  end

  analysis_callback.start_time = time_ns()
  analysis_callback(integrator)
  return nothing
end


# TODO: Taal refactor, allow passing an IO object (which could be devnull to avoid cluttering the console)
function (analysis_callback::AnalysisCallback)(integrator)
  semi = integrator.p
  mesh, equations, solver, cache = mesh_equations_solver_cache(semi)
  @unpack dt, t = integrator
  iter = integrator.destats.naccept

  runtime_absolute = 1.0e-9 * (time_ns() - analysis_callback.start_time)
  runtime_relative = 1.0e-9 * take!(semi.performance_counter) / ndofs(semi)

  @trixi_timeit timer() "analyze solution" begin
    # General information
    mpi_println()
    mpi_println("─"^100)
    # TODO: Taal refactor, polydeg is specific to DGSEM
    mpi_println(" Simulation running '", get_name(equations), "' with ", summary(solver))
    mpi_println("─"^100)
    mpi_println(" #timesteps:     " * @sprintf("% 14d", iter) *
                "               " *
                " run time:       " * @sprintf("%10.8e s", runtime_absolute))
    mpi_println(" Δt:             " * @sprintf("%10.8e", dt) *
                "               " *
                " time/DOF/rhs!:  " * @sprintf("%10.8e s", runtime_relative))
    mpi_println(" sim. time:      " * @sprintf("%10.8e", t))
    mpi_println(" #DOF:           " * @sprintf("% 14d", ndofs(semi)))
    mpi_println(" #elements:      " * @sprintf("% 14d", nelements(mesh, solver, cache)))

    # Level information (only show for AMR)
    print_amr_information(integrator.opts.callback, mesh, solver, cache)
    mpi_println()

    # Open file for appending and store time step and time information
    if mpi_isroot() && analysis_callback.save_analysis
      io = open(joinpath(analysis_callback.output_directory, analysis_callback.analysis_filename), "a")
      @printf(io, "% 9d", iter)
      @printf(io, "  %10.8e", t)
      @printf(io, "  %10.8e", dt)
    else
      io = devnull
    end

    # Calculate current time derivative (needed for semidiscrete entropy time derivative, residual, etc.)
    du_ode = first(get_tmp_cache(integrator))
    @notimeit timer() rhs!(du_ode, integrator.u, semi, t)
    u  = wrap_array(integrator.u, mesh, equations, solver, cache)
    du = wrap_array(du_ode,       mesh, equations, solver, cache)
    l2_error, linf_error = analysis_callback(io, du, u, integrator.u, t, semi)

    mpi_println("─"^100)
    mpi_println()

    # Add line break and close analysis file if it was opened
    if mpi_isroot() && analysis_callback.save_analysis
      # This resolves a possible type instability introduced above, since `io`
      # can either be an `IOStream` or `devnull`, but we know that it must be
      # an `IOStream here`.
      println(io::IOStream)
      close(io::IOStream)
    end
  end

  # avoid re-evaluating possible FSAL stages
  u_modified!(integrator, false)

  # Return errors for EOC analysis
  return l2_error, linf_error
end


# This method is just called internally from `(analysis_callback::AnalysisCallback)(integrator)`
# and serves as a function barrier. Additionally, it makes the code easier to profile and optimize.
function (analysis_callback::AnalysisCallback)(io, du, u, u_ode, t, semi)
  @unpack analyzer, analysis_errors, analysis_integrals = analysis_callback
  cache_analysis = analysis_callback.cache
  _, equations, _, _ = mesh_equations_solver_cache(semi)

  # Calculate and print derived quantities (error norms, entropy etc.)
  # Variable names required for L2 error, Linf error, and conservation error
  if any(q in analysis_errors for q in
         (:l2_error, :linf_error, :conservation_error, :residual)) && mpi_isroot()
    print(" Variable:    ")
    for v in eachvariable(equations)
      @printf("   %-14s", varnames(cons2cons, equations)[v])
    end
    println()
  end

  # Calculate L2/Linf errors, which are also returned
  l2_error, linf_error = calc_error_norms(u_ode, t, analyzer, semi, cache_analysis)

  if mpi_isroot()
    # L2 error
    if :l2_error in analysis_errors
      print(" L2 error:    ")
      for v in eachvariable(equations)
        @printf("  % 10.8e", l2_error[v])
        @printf(io, "  % 10.8e", l2_error[v])
      end
      println()
    end

    # Linf error
    if :linf_error in analysis_errors
      print(" Linf error:  ")
      for v in eachvariable(equations)
        @printf("  % 10.8e", linf_error[v])
        @printf(io, "  % 10.8e", linf_error[v])
      end
      println()
    end
  end


  # Conservation errror
  if :conservation_error in analysis_errors
    @unpack initial_state_integrals = analysis_callback
    state_integrals = integrate(u_ode, semi)

    if mpi_isroot()
      print(" |∑U - ∑U₀|:  ")
      for v in eachvariable(equations)
        err = abs(state_integrals[v] - initial_state_integrals[v])
        @printf("  % 10.8e", err)
        @printf(io, "  % 10.8e", err)
      end
      println()
    end
  end

  # Residual (defined here as the vector maximum of the absolute values of the time derivatives)
  if :residual in analysis_errors
    mpi_print(" max(|Uₜ|):   ")
    for v in eachvariable(equations)
      # Calculate maximum absolute value of Uₜ
      res = maximum(abs, view(du, v, ..))
      if mpi_isparallel()
        # TODO: Debugging, here is a type instability
        global_res = MPI.Reduce!(Ref(res), max, mpi_root(), mpi_comm())
        if mpi_isroot()
          res::eltype(du) = global_res[]
        end
      end
      if mpi_isroot()
        @printf("  % 10.8e", res)
        @printf(io, "  % 10.8e", res)
      end
    end
    mpi_println()
  end

  # L2/L∞ errors of the primitive variables
  if :l2_error_primitive in analysis_errors || :linf_error_primitive in analysis_errors
    l2_error_prim, linf_error_prim = calc_error_norms(cons2prim, u_ode, t, analyzer, semi, cache_analysis)

    if mpi_isroot()
      print(" Variable:    ")
      for v in eachvariable(equations)
        @printf("   %-14s", varnames(cons2prim, equations)[v])
      end
      println()

      # L2 error
      if :l2_error_primitive in analysis_errors
        print(" L2 error prim.: ")
        for v in eachvariable(equations)
          @printf("%10.8e   ", l2_error_prim[v])
          @printf(io, "  % 10.8e", l2_error_prim[v])
        end
        println()
      end

      # L∞ error
      if :linf_error_primitive in analysis_errors
        print(" Linf error pri.:")
        for v in eachvariable(equations)
          @printf("%10.8e   ", linf_error_prim[v])
          @printf(io, "  % 10.8e", linf_error_prim[v])
        end
        println()
      end
    end
  end

  # additional integrals
  analyze_integrals(analysis_integrals, io, du, u, t, semi)

  return l2_error, linf_error
end


# Print level information only if AMR is enabled
function print_amr_information(callbacks, mesh, solver, cache)

  # Return early if there is nothing to print
  uses_amr(callbacks) || return nothing

  levels = Vector{Int}(undef, nelements(solver, cache))
  min_level = typemax(Int)
  max_level = typemin(Int)
  for element in eachelement(solver, cache)
    current_level = mesh.tree.levels[cache.elements.cell_ids[element]]
    levels[element] = current_level
    min_level = min(min_level, current_level)
    max_level = max(max_level, current_level)
  end

  for level = max_level:-1:min_level+1
    mpi_println(" ├── level $level:    " * @sprintf("% 14d", count(==(level), levels)))
  end
  mpi_println(" └── level $min_level:    " * @sprintf("% 14d", count(==(min_level), levels)))

  return nothing
end

# Print level information only if AMR is enabled
function print_amr_information(callbacks, mesh::P4estMesh, solver, cache)

  # Return early if there is nothing to print
  uses_amr(callbacks) || return nothing

  elements_per_level = zeros(P4EST_MAXLEVEL + 1)

  for tree in unsafe_wrap_sc(p4est_tree_t, mesh.p4est.trees)
    elements_per_level .+= tree.quadrants_per_level
  end

  min_level = findfirst(i -> i > 0, elements_per_level) - 1
  max_level = findlast(i -> i > 0, elements_per_level) - 1

  for level = max_level:-1:min_level+1
    mpi_println(" ├── level $level:    " * @sprintf("% 14d", elements_per_level[level + 1]))
  end
  mpi_println(" └── level $min_level:    " * @sprintf("% 14d", elements_per_level[min_level + 1]))

  return nothing
end


# Iterate over tuples of analysis integrals in a type-stable way using "lispy tuple programming".
function analyze_integrals(analysis_integrals::NTuple{N,Any}, io, du, u, t, semi) where {N}

  # Extract the first analysis integral and process it; keep the remaining to be processed later
  quantity = first(analysis_integrals)
  remaining_quantities = Base.tail(analysis_integrals)

  res = analyze(quantity, du, u, t, semi)
  if mpi_isroot()
    @printf(" %-12s:", pretty_form_utf(quantity))
    @printf("  % 10.8e", res)
    @printf(io, "  % 10.8e", res)
  end
  mpi_println()

  # Recursively call this method with the unprocessed integrals
  analyze_integrals(remaining_quantities, io, du, u, t, semi)
  return nothing
end

# terminate the type-stable iteration over tuples
function analyze_integrals(analysis_integrals::Tuple{}, io, du, u, t, semi)
  nothing
end


# used for error checks and EOC analysis
function (cb::DiscreteCallback{Condition,Affect!})(sol) where {Condition, Affect!<:AnalysisCallback}
  analysis_callback = cb.affect!
  semi = sol.prob.p
  @unpack analyzer = analysis_callback
  cache_analysis = analysis_callback.cache

  l2_error, linf_error = calc_error_norms(sol.u[end], sol.t[end], analyzer, semi, cache_analysis)
  (; l2=l2_error, linf=linf_error)
end


# some common analysis_integrals
# to support another analysis integral, you can overload
# Trixi.analyze, Trixi.pretty_form_utf, Trixi.pretty_form_ascii
function analyze(quantity, du, u, t, semi::AbstractSemidiscretization)
  mesh, equations, solver, cache = mesh_equations_solver_cache(semi)
  analyze(quantity, du, u, t, mesh, equations, solver, cache)
end
function analyze(quantity, du, u, t, mesh, equations, solver, cache)
  integrate(quantity, u, mesh, equations, solver, cache, normalize=true)
end
pretty_form_utf(quantity) = get_name(quantity)
pretty_form_ascii(quantity) = get_name(quantity)


function entropy_timederivative end
pretty_form_utf(::typeof(entropy_timederivative)) = "∑∂S/∂U ⋅ Uₜ"
pretty_form_ascii(::typeof(entropy_timederivative)) = "dsdu_ut"

pretty_form_utf(::typeof(entropy)) = "∑S"

pretty_form_utf(::typeof(energy_total)) = "∑e_total"
pretty_form_ascii(::typeof(energy_total)) = "e_total"

pretty_form_utf(::typeof(energy_kinetic)) = "∑e_kinetic"
pretty_form_ascii(::typeof(energy_kinetic)) = "e_kinetic"

pretty_form_utf(::typeof(energy_kinetic_nondimensional)) = "∑e_kinetic*"
pretty_form_ascii(::typeof(energy_kinetic_nondimensional)) = "e_kinetic*"

pretty_form_utf(::typeof(energy_internal)) = "∑e_internal"
pretty_form_ascii(::typeof(energy_internal)) = "e_internal"

pretty_form_utf(::typeof(energy_magnetic)) = "∑e_magnetic"
pretty_form_ascii(::typeof(energy_magnetic)) = "e_magnetic"

pretty_form_utf(::typeof(cross_helicity)) = "∑v⋅B"
pretty_form_ascii(::typeof(cross_helicity)) = "v_dot_B"

pretty_form_utf(::Val{:l2_divb}) = "L2 ∇⋅B"
pretty_form_ascii(::Val{:l2_divb}) = "l2_divb"

pretty_form_utf(::Val{:linf_divb}) = "L∞ ∇⋅B"
pretty_form_ascii(::Val{:linf_divb}) = "linf_divb"

pretty_form_utf(::typeof(lake_at_rest_error)) = "∑|H₀-(h+b)|"
pretty_form_ascii(::typeof(lake_at_rest_error)) = "|H0-(h+b)|"

# specialized implementations specific to some solvers
include("analysis_dg1d.jl")
include("analysis_dg2d.jl")
include("analysis_dg2d_parallel.jl")
include("analysis_dg3d.jl")


end # @muladd
