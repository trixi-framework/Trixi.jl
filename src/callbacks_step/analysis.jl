
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
mutable struct AnalysisCallback{Analyzer<:SolutionAnalyzer, AnalysisIntegrals, InitialStateIntegrals}
  start_time::Float64
  interval::Int
  save_analysis::Bool
  output_directory::String
  analysis_filename::String
  analyzer::Analyzer
  analysis_errors::Vector{Symbol}
  analysis_integrals::AnalysisIntegrals
  initial_state_integrals::InitialStateIntegrals
end


# TODO: Taal bikeshedding, implement a method with less information and the signature
# function Base.show(io::IO, analysis_callback::AnalysisCallback)
# end
function Base.show(io::IO, ::MIME"text/plain", cb::DiscreteCallback{Condition,Affect!}) where {Condition, Affect!<:AnalysisCallback}
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
  _, equations, solver, _ = mesh_equations_solver_cache(semi)
  AnalysisCallback(equations, solver; kwargs...)
end

function AnalysisCallback(equations::AbstractEquations, solver;
                          interval=0,
                          save_analysis=false,
                          output_directory="out",
                          analysis_filename="analysis.dat",
                          extra_analysis_errors=Symbol[],
                          analysis_errors=union(default_analysis_errors(equations), extra_analysis_errors),
                          extra_analysis_integrals=(),
                          analysis_integrals=union(default_analysis_integrals(equations), extra_analysis_integrals),
                          kwargs...)
  # when is the callback activated
  condition = (u, t, integrator) -> interval > 0 && (integrator.iter % interval == 0 ||
                                                     isfinished(integrator))

  analysis_callback = AnalysisCallback(0.0, interval, save_analysis, output_directory, analysis_filename,
                                       SolutionAnalyzer(solver; kwargs...),
                                       analysis_errors, Tuple(analysis_integrals),
                                       SVector(ntuple(_ -> zero(real(solver)), nvariables(equations))))

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
        for v in varnames_cons(equations)
          @printf(io, "   %-14s", "l2_" * v)
        end
      end
      if :linf_error in analysis_errors
        for v in varnames_cons(equations)
          @printf(io, "   %-14s", "linf_" * v)
        end
      end
      if :conservation_error in analysis_errors
        for v in varnames_cons(equations)
          @printf(io, "   %-14s", "cons_" * v)
        end
      end
      if :residual in analysis_errors
        for v in varnames_cons(equations)
          @printf(io, "   %-14s", "res_" * v)
        end
      end
      if :l2_error_primitive in analysis_errors
        for v in varnames_prim(equations)
          @printf(io, "   %-14s", "l2_" * v)
        end
      end
      if :linf_error_primitive in analysis_errors
        for v in varnames_prim(equations)
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
  @unpack analyzer, analysis_errors, analysis_integrals = analysis_callback
  @unpack dt, t, iter = integrator
  u = wrap_array(integrator.u, mesh, equations, solver, cache)

  runtime_absolute = 1.0e-9 * (time_ns() - analysis_callback.start_time)
  runtime_relative = 1.0e-9 * take!(semi.performance_counter) / ndofs(semi)

  @timeit_debug timer() "analyze solution" begin
    # General information
    mpi_println()
    mpi_println("─"^100)
    # TODO: Taal refactor, polydeg is specific to DGSEM
    mpi_println(" Simulation running '", get_name(equations), "' with polydeg = ", polydeg(solver))
    mpi_println("─"^100)
    mpi_println(" #timesteps:     " * @sprintf("% 14d", iter) *
                "               " *
                " run time:       " * @sprintf("%10.8e s", runtime_absolute))
    mpi_println(" Δt:             " * @sprintf("%10.8e", dt) *
                "               " *
                " time/DOF/rhs!:  " * @sprintf("%10.8e s", runtime_relative))
    mpi_println(" sim. time:      " * @sprintf("%10.8e", t))
    mpi_println(" #DOF:           " * @sprintf("% 14d", ndofs(semi)))
    mpi_println(" #elements:      " * @sprintf("% 14d", nelements(solver, cache)))

    # Level information (only show for AMR)
    uses_amr = false
    callbacks = integrator.opts.callback
    if callbacks isa CallbackSet
      for cb in callbacks.discrete_callbacks
        if cb.affect! isa AMRCallback
          uses_amr = true
          break
        end
      end
    end
    if uses_amr
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
        mpi_println(" ├── level $level:    " * @sprintf("% 14d", count(isequal(level), levels)))
      end
      mpi_println(" └── level $min_level:    " * @sprintf("% 14d", count(isequal(min_level), levels)))
    end
    mpi_println()

    # Open file for appending and store time step and time information
    if mpi_isroot() && analysis_callback.save_analysis
      io = open(joinpath(analysis_callback.output_directory, analysis_callback.analysis_filename), "a")
      mpi_isroot() && @printf(io, "% 9d", iter)
      mpi_isroot() && @printf(io, "  %10.8e", t)
      mpi_isroot() && @printf(io, "  %10.8e", dt)
    end

    # the time derivative can be unassigned before the first step is made
    if t == integrator.sol.prob.tspan[1]
      du_ode = similar(integrator.u)
    else
      du_ode = get_du(integrator)
    end
    @notimeit timer() rhs!(du_ode, integrator.u, semi, t)
    GC.@preserve du_ode begin
      du = wrap_array(du_ode, mesh, equations, solver, cache)

      # Calculate and print derived quantities (error norms, entropy etc.)
      # Variable names required for L2 error, Linf error, and conservation error
      if any(q in analysis_errors for q in
            (:l2_error, :linf_error, :conservation_error, :residual))
        mpi_print(" Variable:    ")
        for v in eachvariable(equations)
          mpi_isroot() && @printf("   %-14s", varnames_cons(equations)[v])
        end
        mpi_println()
      end

      # Calculate L2/Linf errors, which are also returned by analyze_solution
      l2_error, linf_error = calc_error_norms(u, t, analyzer, semi)

      # L2 error
      if :l2_error in analysis_errors
        mpi_print(" L2 error:    ")
        for v in eachvariable(equations)
          mpi_isroot() && @printf("  % 10.8e", l2_error[v])
          analysis_callback.save_analysis && mpi_isroot() && @printf(io, "  % 10.8e", l2_error[v])
        end
        mpi_println()
      end

      # Linf error
      if :linf_error in analysis_errors
        mpi_print(" Linf error:  ")
        for v in eachvariable(equations)
          mpi_isroot() && @printf("  % 10.8e", linf_error[v])
          analysis_callback.save_analysis && mpi_isroot() && @printf(io, "  % 10.8e", linf_error[v])
        end
        mpi_println()
      end

      # Conservation errror
      if :conservation_error in analysis_errors
        @unpack initial_state_integrals = analysis_callback
        state_integrals = integrate(integrator.u, semi)

        mpi_print(" |∑U - ∑U₀|:  ")
        for v in eachvariable(equations)
          err = abs(state_integrals[v] - initial_state_integrals[v])
          mpi_isroot() && @printf("  % 10.8e", err)
          analysis_callback.save_analysis && mpi_isroot() && @printf(io, "  % 10.8e", err)
        end
        mpi_println()
      end

      # Residual (defined here as the vector maximum of the absolute values of the time derivatives)
      if :residual in analysis_errors
        mpi_print(" max(|Uₜ|):   ")
        for v in eachvariable(equations)
          # Calculate maximum absolute value of Uₜ
          res = maximum(abs, view(du, v, ..))
          if mpi_isparallel()
            global_res = MPI.Reduce!(Ref(res), max, mpi_root(), mpi_comm())
            if mpi_isroot()
              res = global_res[]
            end
          end
          mpi_isroot() && @printf("  % 10.8e", res)
          analysis_callback.save_analysis && mpi_isroot() && @printf(io, "  % 10.8e", res)
        end
        mpi_println()
      end

      # L2/L∞ errors of the primitive variables
      if :l2_error_primitive in analysis_errors || :linf_error_primitive in analysis_errors
        l2_error_prim, linf_error_prim = calc_error_norms(cons2prim, u, t, analyzer, semi)

        mpi_print(" Variable:    ")
        for v in eachvariable(equations)
          mpi_isroot() && @printf("   %-14s", varnames_prim(equations)[v])
        end
        mpi_println()

        # L2 error
        if :l2_error_primitive in analysis_errors
          mpi_print(" L2 error prim.: ")
          for v in eachvariable(equations)
            mpi_isroot() && @printf("%10.8e   ", l2_error_prim[v])
            analysis_callback.save_analysis && mpi_isroot() && @printf(io, "  % 10.8e", l2_error_prim[v])
          end
          mpi_println()
        end

        # L∞ error
        if :linf_error_primitive in analysis_errors
          mpi_print(" Linf error pri.:")
          for v in eachvariable(equations)
            mpi_isroot() && @printf("%10.8e   ", linf_error_prim[v])
            analysis_callback.save_analysis && mpi_isroot() && @printf(io, "  % 10.8e", linf_error_prim[v])
          end
          mpi_println()
        end
      end


      # additional
      for quantity in analysis_integrals
        res = analyze(quantity, du, u, t, semi)
        mpi_isroot() && @printf(" %-12s:", pretty_form_utf(quantity))
        mpi_isroot() && @printf("  % 10.8e", res)
        analysis_callback.save_analysis && mpi_isroot() && @printf(io, "  % 10.8e", res)
        mpi_println()
      end
    end # GC.@preserve du_ode

    mpi_println("─"^100)
    mpi_println()

    # Add line break and close analysis file if it was opened
    if mpi_isroot() && analysis_callback.save_analysis
      println(io)
      close(io)
    end
  end

  # avoid re-evaluating possible FSAL stages
  u_modified!(integrator, false)

  # Return errors for EOC analysis
  return l2_error, linf_error
end


# used for error checks and EOC analysis
function (cb::DiscreteCallback{Condition,Affect!})(sol) where {Condition, Affect!<:AnalysisCallback}
  analysis_callback = cb.affect!
  semi = sol.prob.p
  @unpack analyzer = analysis_callback

  l2_error, linf_error = calc_error_norms(sol.u[end], sol.t[end], analyzer, semi)
  (; l2=l2_error, linf=linf_error)
end


# some common analysis_integrals
# to support another analysis integral, you can overload
# Trixi.analyze, Trixi.pretty_form_utf, Trixi.pretty_form_ascii
@inline function analyze(quantity, du, u, t, semi::AbstractSemidiscretization)
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


# specialized implementations specific to some solvers
include("analysis_dg1d.jl")
include("analysis_dg2d.jl")
include("analysis_dg2d_parallel.jl")
include("analysis_dg3d.jl")
