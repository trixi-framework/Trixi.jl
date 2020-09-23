
# TODO: Taal refactor
# - analysis_interval part as PeriodicCallback called after a certain amount of simulation time
mutable struct AnalysisCallback{Analyzer<:SolutionAnalyzer, AnalysisIntegrals, InitialStateIntegrals}
  start_time::Float64
  save_analysis::Bool
  analysis_filename::String
  analyzer::Analyzer
  analysis_errors::Vector{Symbol}
  analysis_integrals::AnalysisIntegrals
  initial_state_integrals::InitialStateIntegrals
end

function AnalysisCallback(semi::SemidiscretizationHyperbolic;
                          analysis_interval=0,
                          save_analysis=false, analysis_filename="analysis.dat",
                          extra_analysis_errors=Symbol[],
                          analysis_errors=union(default_analysis_errors(semi.equations), extra_analysis_errors),
                          extra_analysis_integrals=(),
                          analysis_integrals=union(default_analysis_integrals(semi.equations), extra_analysis_integrals))
  # when is the callback activated
  condition = (u, t, integrator) -> analysis_interval > 0 && (integrator.iter % analysis_interval == 0 || t in integrator.sol.prob.tspan)

  @unpack equations, solver = semi
  analysis_callback = AnalysisCallback(0.0, save_analysis, analysis_filename, SolutionAnalyzer(solver),
                                       analysis_errors, Tuple(analysis_integrals),
                                       SVector(ntuple(_ -> zero(real(solver)), nvariables(equations))))

  DiscreteCallback(condition, analysis_callback,
                   save_positions=(false,false),
                   initialize=initialize!)
end

function initialize!(cb::DiscreteCallback{Condition,Affect!}, u, t, integrator) where {Condition, Affect!<:AnalysisCallback}
  semi = integrator.p
  initial_state_integrals = integrate(u, semi)

  analysis_callback = cb.affect!
  analysis_callback.initial_state_integrals = initial_state_integrals
  @unpack save_analysis, analysis_filename, analysis_errors, analysis_integrals = analysis_callback

  # write header of output file
  save_analysis && open(analysis_filename, "w") do io
    @printf(io, "#%-8s", "timestep")
    @printf(io, "  %-14s", "time")
    @printf(io, "  %-14s", "dt")
    if :l2_error in analysis_errors
      for v in varnames_cons(semi.equations)
        @printf(io, "   %-14s", "l2_" * v)
      end
    end
    if :linf_error in analysis_errors
      for v in varnames_cons(semi.equations)
        @printf(io, "   %-14s", "linf_" * v)
      end
    end
    if :conservation_error in analysis_errors
      for v in varnames_cons(semi.equations)
        @printf(io, "   %-14s", "cons_" * v)
      end
    end
    if :residual in analysis_errors
      for v in varnames_cons(semi.equations)
        @printf(io, "   %-14s", "res_" * v)
      end
    end
    if :l2_error_primitive in analysis_errors
      for v in varnames_prim(semi.equations)
        @printf(io, "   %-14s", "l2_" * v)
      end
    end
    if :linf_error_primitive in analysis_errors
      for v in varnames_prim(semi.equations)
        @printf(io, "   %-14s", "linf_" * v)
      end
    end

    for quantity in analysis_integrals
      @printf(io, "   %-14s", pretty_form_file(quantity))
    end

    println(io)
  end

  analysis_callback.start_time = time_ns()
  analysis_callback(integrator)
  return nothing
end


# TODO: Taal bikeshedding, implement a method with less information and the signature
# function Base.show(io::IO, analysis_callback::AnalysisCallback)
# end
function Base.show(io::IO, ::MIME"text/plain", cb::DiscreteCallback{Condition,Affect!}) where {Condition, Affect!<:AnalysisCallback}
  analysis_callback = cb.affect!
  @unpack save_analysis, analysis_filename, analyzer, analysis_errors, analysis_integrals = analysis_callback
  println(io, "AnalysisCallback with")
  println(io, "- save_analysis: ", save_analysis)
  println(io, "- analysis_filename: ", analysis_filename)
  println(io, "- analyzer: ", analyzer)
  println(io, "- analysis_errors: ", analysis_errors)
  print(io,   "- analysis_integrals: ", analysis_integrals)
end


# TODO: Taal refactor, allow passing an IO object (which could be devnull to avoid cluttering the console)
function (analysis_callback::AnalysisCallback)(integrator)
  semi = integrator.p
  @unpack mesh, equations, solver, cache = semi
  @unpack analyzer, analysis_errors, analysis_integrals = analysis_callback
  @unpack dt, t, iter = integrator
  u = wrap_array(integrator.u, mesh, equations, solver, cache)

  runtime_absolute = 1.0e-9 * (time_ns() - analysis_callback.start_time)
  runtime_relative = 1.0e-9 * take!(semi.performance_counter) / ndofs(semi)

  @timeit_debug timer() "analyze solution" begin
    # General information
    println()
    println("-"^80)
    # TODO: Taal refactor, polydeg is specific to DGSEM
    println(" Simulation running '", get_name(equations), "' with POLYDEG = ", polydeg(solver))
    println("-"^80)
    println(" #timesteps:     " * @sprintf("% 14d", iter) *
            "               " *
            " run time:       " * @sprintf("%10.8e s", runtime_absolute))
    println(" dt:             " * @sprintf("%10.8e", dt) *
            "               " *
            " Time/DOF/rhs!:  " * @sprintf("%10.8e s", runtime_relative))
    println(" sim. time:      " * @sprintf("%10.8e", t))

    # TODO: Taal refactor, what to do with output and AMR?
    # Level information (only show for AMR)
    # if parameter("amr_interval", 0)::Int > 0
    #   levels = Vector{Int}(undef, dg.n_elements)
    #   for element_id in 1:dg.n_elements
    #     levels[element_id] = mesh.tree.levels[dg.elements.cell_ids[element_id]]
    #   end
    #   min_level = minimum(levels)
    #   max_level = maximum(levels)

    #   println(" #elements:      " * @sprintf("% 14d", dg.n_elements))
    #   for level = max_level:-1:min_level+1
    #     println(" ├── level $level:    " * @sprintf("% 14d", count(x->x==level, levels)))
    #   end
    #   println(" └── level $min_level:    " * @sprintf("% 14d", count(x->x==min_level, levels)))
    # end
    println()

    # Open file for appending and store time step and time information
    if analysis_callback.save_analysis
      io = open(analysis_callback.analysis_filename, "a")
      @printf(io, "% 9d", step)
      @printf(io, "  %10.8e", time)
      @printf(io, "  %10.8e", dt)
    end

    # the time derivative can be unassigned before the first step is made
    if t == integrator.sol.prob.tspan[1]
      u_vector  = integrator.u
      du_vector = similar(u_vector)
      @notimeit timer() rhs!(du_vector, u_vector, semi, t)
    else
      du_vector = get_du(integrator)
    end
    du = wrap_array(du_vector, mesh, equations, solver, cache)

    # Calculate and print derived quantities (error norms, entropy etc.)
    # Variable names required for L2 error, Linf error, and conservation error
    if any(q in analysis_errors for q in
          (:l2_error, :linf_error, :conservation_error, :residual))
      print(" Variable:    ")
      for v in eachvariable(equations)
        @printf("   %-14s", varnames_cons(equations)[v])
      end
      println()
    end

    # Calculate L2/Linf errors, which are also returned by analyze_solution
    l2_error, linf_error = calc_error_norms(u, t, analyzer, semi)

    # L2 error
    if :l2_error in analysis_errors
      print(" L2 error:    ")
      for v in eachvariable(equations)
        @printf("  % 10.8e", l2_error[v])
        analysis_callback.save_analysis && @printf(io, "  % 10.8e", l2_error[v])
      end
      println()
    end

    # Linf error
    if :linf_error in analysis_errors
      print(" Linf error:  ")
      for v in eachvariable(equations)
        @printf("  % 10.8e", linf_error[v])
        analysis_callback.save_analysis && @printf(io, "  % 10.8e", linf_error[v])
      end
      println()
    end

    # Conservation errror
    if :conservation_error in analysis_errors
      @unpack analysis_callback = analysis_callback
      state_integrals = integrate(u, semi)

      print(" |∑U - ∑U₀|:  ")
      for v in eachvariable(equations)
        err = abs(state_integrals[v] - initial_state_integrals[v])
        @printf("  % 10.8e", err)
        analysis_callback.save_analysis && @printf(io, "  % 10.8e", err)
      end
      println()
    end

    # Residual (defined here as the vector maximum of the absolute values of the time derivatives)
    if :residual in analysis_errors
      print(" max(|Uₜ|):   ")
      for v in eachvariable(equations)
        # Calculate maximum absolute value of Uₜ
        @views res = maximum(abs, view(du, v, ..))
        @printf("  % 10.8e", res)
        analysis_callback.save_analysis && @printf(io, "  % 10.8e", res)
      end
      println()
    end

    # L2/L∞ errors of the primitive variables
    if :l2_error_primitive in analysis_errors || :linf_error_primitive in analysis_errors
      l2_error_prim, linf_error_prim = calc_error_norms(cons2prim, semi, t)

      print(" Variable:    ")
      for v in eachvariable(equations)
        @printf("   %-14s", varnames_prim(equations)[v])
      end
      println()

      # L2 error
      if :l2_error_primitive in analysis_errors
        print(" L2 error prim.: ")
        for v in eachvariable(equations)
          @printf("%10.8e   ", l2_error_prim[v])
          analysis_callback.save_analysis && @printf(io, "  % 10.8e", l2_error_prim[v])
        end
        println()
      end

      # L∞ error
      if :linf_error_primitive in analysis_errors
        print(" Linf error pri.:")
        for v in eachvariable(equations)
          @printf("%10.8e   ", linf_error_prim[v])
          analysis_callback.save_analysis && @printf(io, "  % 10.8e", linf_error_prim[v])
        end
        println()
      end
    end


    # additional
    for quantity in analysis_integrals
      res = analyze(quantity, du, u, t, mesh, equations, solver, cache)
      @printf(" %-12s:", pretty_form_repl(quantity))
      @printf("  % 10.8e", res)
      analysis_callback.save_analysis && @printf(io, "  % 10.8e", res)
      println()
    end

    println("-"^80)
    println()

    # Add line break and close analysis file if it was opened
    if analysis_callback.save_analysis
      println(io)
      close(io)
    end
  end

  # Return errors for EOC analysis
  return l2_error, linf_error
end


# used for error checks and EOC analysis
function (cb::DiscreteCallback{Condition,Affect!})(sol::ODESolution) where {Condition, Affect!<:AnalysisCallback}
  analysis_callback = cb.affect!
  semi = sol.prob.p
  @unpack analyzer = analysis_callback

  l2_error, linf_error = calc_error_norms(sol.u[end], sol.t[end], analyzer, semi)
end


# some common analysis_integrals
# to support another analysis integral, you can overload
# Trixi.analyze, Trixi.pretty_form_repl, Trixi.pretty_form_file
function analyze(quantity, du, u, t, mesh, equations, solver, cache)
  integrate(quantity, u, mesh, equations, solver, cache, normalize=true)
end
pretty_form_repl(quantity) = get_name(quantity)
pretty_form_file(quantity) = get_name(quantity)


function entropy_timederivative end
pretty_form_repl(::typeof(entropy_timederivative)) = "∑∂S/∂U ⋅ Uₜ"
pretty_form_file(::typeof(entropy_timederivative)) = "dsdu_ut"

pretty_form_repl(::typeof(entropy)) = "∑S"

pretty_form_repl(::typeof(energy_total)) = "∑e_total"
pretty_form_file(::typeof(energy_total)) = "e_total"


# specialized implementations specific to some solvers
include("analysis_dg2d.jl")
