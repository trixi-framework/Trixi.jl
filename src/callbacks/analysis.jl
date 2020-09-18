
mutable struct AliveCallback
  start_time::Float64
end

function (alive_callback::AliveCallback)(integrator)
  if integrator.t == integrator.sol.prob.tspan[2]
    println("-"^80)
    println("Trixi simulation run finished.    Final time: ", integrator.t, "    Time steps: ", integrator.iter)
    println("-"^80)
    println()

    print_timer(timer(), title="Trixi.jl",
                allocations=true, linechars=:ascii, compact=false)
    println()
  else
    @unpack t, dt, iter = integrator
    runtime_absolute = 1.0e-9 * (time_ns() - alive_callback.start_time)
    @printf("#t/s: %6d | dt: %.4e | Sim. time: %.4e | Run time: %.4e s\n",
            iter, dt, t, runtime_absolute)
  end

  return nothing
end

function AliveCallback(; analysis_interval=0,
                         alive_interval=analysis_interval÷10)
  condition = (u, t, integrator) -> alive_interval > 0 && ((integrator.iter % alive_interval == 0 && integrator.iter % analysis_interval != 0) || t == integrator.sol.prob.tspan[2])

  alive_callback = AliveCallback(0.0)

  DiscreteCallback(condition, alive_callback,
                   save_positions=(false,false),
                   initialize = (c, u, t, integrator) -> begin
                     reset_timer!(timer())
                     c.affect!.start_time = time_ns()
                   end)
end

function initialize!(cb::DiscreteCallback{Condition,Affect!}, u, t, integrator) where {Condition, Affect!<:AliveCallback}
  reset_timer!(timer())
  alive_callback = cb.affect!
  alive_callback.start_time = time_ns()
  return nothing
end


# TODO: Taal refactor
# - analysis_interval part as PeriodicCallback called after a certain amount of simulation time
mutable struct AnalysisCallback{Analyzer<:SolutionAnalyzer, InitialStateIntegrals}
  start_time::Float64
  save_analysis::Bool
  analysis_filename::String
  analysis_quantities::Vector{Symbol}
  analyzer::Analyzer
  initial_state_integrals::InitialStateIntegrals
end

function AnalysisCallback(semi::Semidiscretization;
                          analysis_interval=0,
                          save_analysis=false, analysis_filename="analysis.dat",
                          extra_analysis_quantities=Symbol[])
  # when is the callback activated
  condition = (u, t, integrator) -> analysis_interval > 0 && (integrator.iter % analysis_interval == 0 || t in integrator.sol.prob.tspan)

  @unpack equations, solver = semi
  analysis_quantities = vcat(collect(Symbol.(default_analysis_quantities(equations))),
                             extra_analysis_quantities)
  analysis_callback = AnalysisCallback(0.0, save_analysis, analysis_filename, analysis_quantities,
                                       SolutionAnalyzer(solver),
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
  analysis_callback.start_time = time_ns()
  analysis_callback(integrator)
  return nothing
end


# TODO: Taal bikeshedding, implement a method with less information and the signature
# function Base.show(io::IO, analysis_callback::AnalysisCallback)
# end
function Base.show(io::IO, ::MIME"text/plain", cb::DiscreteCallback{Condition,Affect!}) where {Condition, Affect!<:AnalysisCallback}
  analysis_callback = cb.affect!
  @unpack save_analysis, analysis_filename, analysis_quantities, analyzer = analysis_callback
  println(io, "AnalysisCallback with")
  println(io, "- save_analysis: ", save_analysis)
  println(io, "- analysis_filename: ", analysis_filename)
  println(io, "- analysis_quantities: ", analysis_quantities)
  println(io, "- analyzer: ", analyzer)
end


function (analysis_callback::AnalysisCallback)(integrator)
@timeit_debug timer() "analyze solution" begin
  semi = integrator.p
  @unpack mesh, equations, solver, cache = semi
  @unpack analyzer, analysis_quantities = analysis_callback
  @unpack u, dt, t, iter = integrator

  runtime_absolute = 1.0e-9 * (time_ns() - analysis_callback.start_time)
  runtime_relative = 1.0e-9 * take!(semi.performance_counter) / ndofs(semi)

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

  # TODO: Taal implement, save_analysis
  # Open file for appending and store time step and time information
  if analysis_callback.save_analysis
    io = open(analysis_callback.analysis_filename, "a")
    @printf(io, "% 9d", step)
    @printf(io, "  %10.8e", time)
    @printf(io, "  %10.8e", dt)
  end

  # Calculate and print derived quantities (error norms, entropy etc.)
  # Variable names required for L2 error, Linf error, and conservation error
  if any(q in analysis_quantities for q in
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
  if :l2_error in analysis_quantities
    print(" L2 error:    ")
    for v in eachvariable(equations)
      @printf("  % 10.8e", l2_error[v])
      analysis_callback.save_analysis && @printf(io, "  % 10.8e", l2_error[v])
    end
    println()
  end

  # Linf error
  if :linf_error in analysis_quantities
    print(" Linf error:  ")
    for v in eachvariable(equations)
      @printf("  % 10.8e", linf_error[v])
      analysis_callback.save_analysis && @printf(io, "  % 10.8e", linf_error[v])
    end
    println()
  end

  # Conservation errror
  if :conservation_error in analysis_quantities
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
  if :residual in analysis_quantities
    print(" max(|Uₜ|):   ")
    for v in eachvariable(equations)
      # Calculate maximum absolute value of Uₜ
      @views res = maximum(abs, view(integrator.du, v, ..))
      @printf("  % 10.8e", res)
      analysis_callback.save_analysis && @printf(io, "  % 10.8e", res)
    end
    println()
  end

  # L2/L∞ errors of the primitive variables
  if :l2_error_primitive in analysis_quantities || :linf_error_primitive in analysis_quantities
    l2_error_prim, linf_error_prim = calc_error_norms(cons2prim, semi, t)

    print(" Variable:    ")
    for v in eachvariable(equations)
      @printf("   %-14s", varnames_prim(equations)[v])
    end
    println()

    # L2 error
    if :l2_error_primitive in analysis_quantities
      print(" L2 error prim.: ")
      for v in eachvariable(equations)
        @printf("%10.8e   ", l2_error_prim[v])
        analysis_callback.save_analysis && @printf(io, "  % 10.8e", l2_error_prim[v])
      end
      println()
    end

    # L∞ error
    if :linf_error_primitive in analysis_quantities
      print(" Linf error pri.:")
      for v in eachvariable(equations)
        @printf("%10.8e   ", linf_error_prim[v])
        analysis_callback.save_analysis && @printf(io, "  % 10.8e", linf_error_prim[v])
      end
      println()
    end
  end

  # Entropy time derivative
  if :dsdu_ut in analysis_quantities
    if t == integrator.sol.prob.tspan[1]
      du = similar(u)
      @notimeit timer() rhs!(du, u, semi, t)
    else
      du = get_du(integrator)
    end
    duds_ut = calc_entropy_timederivative(du, u, mesh, equations, solver, cache)
    print(" ∑∂S/∂U ⋅ Uₜ: ")
    @printf("  % 10.8e", duds_ut)
    analysis_callback.save_analysis && @printf(io, "  % 10.8e", duds_ut)
    println()
  end

  # Entropy
  if :entropy in analysis_quantities
    s = integrate(entropy, u, semi)
    print(" ∑S:          ")
    @printf("  % 10.8e", s)
    analysis_callback.save_analysis && @printf(io, "  % 10.8e", s)
    println()
  end

  # Total energy
  if :energy_total in analysis_quantities
    e_total = integrate(energy_total, u, semi)
    print(" ∑e_total:    ")
    @printf("  % 10.8e", e_total)
    analysis_callback.save_analysis && @printf(io, "  % 10.8e", e_total)
    println()
  end

  # TODO: Taal implement additional analysis quantities
  # TODO: Taal refactor, use a tuple of functions that compute the stuff?
  #       We could check the numer of supported arguments and use
  #       integrate(func, u, semi) for func(u_node, equations)
  #       integrate(func, semi, u) for func(u, indices..., equations, solver, args...)

  println("-"^80)
  println()

  # Add line break and close analysis file if it was opened
  if analysis_callback.save_analysis
    println(io)
    close(io)
  end

  # Return errors for EOC analysis
  return l2_error, linf_error
end end

include("analysis_dg2d.jl")
