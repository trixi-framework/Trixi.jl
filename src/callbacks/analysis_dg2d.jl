
# TODO: Taal refactor, split into
# - alive_interval part as DiscreteCallback called every nth iteration
# - analysis_interval part as PeriodicCallback called after a certain amount of simulation time
mutable struct AnalysisCallback{Analyzer<:SolutionAnalyzer}
  start_time::Float64
  save_analysis::Bool
  analysis_filename::String
  analysis_quantities::Vector{Symbol}
  analyzer::Analyzer
end

function AnalysisCallback(semi::Semidiscretization;
                          analysis_interval=0, alive_interval=analysis_interval÷10,
                          save_analysis=false, analysis_filename="analysis.dat",
                          extra_analysis_quantities=Symbol[])
  # when is the callback activated
  condition = (u, t, integrator) -> analysis_interval > 0 && (integrator.iter % analysis_interval == 0 || t == integrator.sol.prob.tspan[2])

  @unpack equations, solver = semi
  analysis_quantities = vcat(collect(Symbol.(default_analysis_quantities(equations))),
                             extra_analysis_quantities)
  analysis_callback = AnalysisCallback(0.0, save_analysis, analysis_filename, analysis_quantities, SolutionAnalyzer(solver))

  DiscreteCallback(condition, analysis_callback,
                   save_positions=(false,false),
                   initialize = (c, u, t, integrator) -> c.affect!.start_time = time_ns())
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
  semi = integrator.p
  @unpack equations, solver = semi
  @unpack analyzer, analysis_quantities = analysis_callback
  @unpack u, dt, t, iter = integrator

  runtime_absolute = 1.0e-9 * (time_ns() - analysis_callback.start_time)
  # TODO: Taal implement runtime_relative
  runtime_relative = NaN

  # General information
  println()
  println("-"^80)
  println(" Simulation running '", get_name(equations), "' with POLYDEG = ", polydeg(solver))
  println("-"^80)
  println(" #timesteps:     " * @sprintf("% 14d", iter) *
          "               " *
          " run time:       " * @sprintf("%10.8e s", runtime_absolute))
  println(" dt:             " * @sprintf("%10.8e", dt) *
          "               " *
          " Time/DOF/step:  " * @sprintf("%10.8e s", runtime_relative))
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
      analysis_callback.save_analysis && @printf(f, "  % 10.8e", l2_error[v])
    end
    println()
  end

  # Linf error
  if :linf_error in analysis_quantities
    print(" Linf error:  ")
    for v in eachvariable(equations)
      @printf("  % 10.8e", linf_error[v])
      analysis_callback.save_analysis && @printf(f, "  % 10.8e", linf_error[v])
    end
    println()
  end

  # TODO: Taal implement additional analysis quantities
  # # Conservation errror
  # if :conservation_error in analysis_quantities
  #   # Calculate state integrals
  #   state_integrals = integrate(dg.elements.u, dg)

  #   # Store initial state integrals at first invocation
  #   if isempty(dg.initial_state_integrals)
  #     dg.initial_state_integrals = zeros(nvariables(equations))
  #     dg.initial_state_integrals .= state_integrals
  #   end

  #   print(" |∑U - ∑U₀|:  ")
  #   for v in eachvariable(equations)
  #     err = abs(state_integrals[v] - dg.initial_state_integrals[v])
  #     @printf("  % 10.8e", err)
  #     analysis_callback.save_analysis && @printf(f, "  % 10.8e", err)
  #   end
  #   println()
  # end

  # TODO: Taal implement additional analysis quantities
  # # Residual (defined here as the vector maximum of the absolute values of the time derivatives)
  # if :residual in analysis_quantities
  #   print(" max(|Uₜ|):   ")
  #   for v in eachvariable(equations)
  #     # Calculate maximum absolute value of Uₜ
  #     @views res = maximum(abs, view(dg.elements.u_t, v, :, :, :))
  #     @printf("  % 10.8e", res)
  #     analysis_callback.save_analysis && @printf(f, "  % 10.8e", res)
  #   end
  #   println()
  # end

  # TODO: Taal implement additional analysis quantities
  # # L2/L∞ errors of the primitive variables
  # if :l2_error_primitive in analysis_quantities || :linf_error_primitive in analysis_quantities
  #   l2_error_prim, linf_error_prim = calc_error_norms(cons2prim, dg, time)

  #   print(" Variable:    ")
  #   for v in eachvariable(equations)
  #     @printf("   %-14s", varnames_prim(equations)[v])
  #   end
  #   println()

  #   # L2 error
  #   if :l2_error_primitive in analysis_quantities
  #     print(" L2 error prim.: ")
  #     for v in eachvariable(equations)
  #       @printf("%10.8e   ", l2_error_prim[v])
  #       analysis_callback.save_analysis && @printf(f, "  % 10.8e", l2_error_prim[v])
  #     end
  #     println()
  #   end

  #   # L∞ error
  #   if :linf_error_primitive in analysis_quantities
  #     print(" Linf error pri.:")
  #     for v in eachvariable(equations)
  #       @printf("%10.8e   ", linf_error_prim[v])
  #       analysis_callback.save_analysis && @printf(f, "  % 10.8e", linf_error_prim[v])
  #     end
  #     println()
  #   end
  # end

  # TODO: Taal implement additional analysis quantities
  # # Entropy time derivative
  # if :dsdu_ut in analysis_quantities
  #   duds_ut = calc_entropy_timederivative(dg, time)
  #   print(" ∑∂S/∂U ⋅ Uₜ: ")
  #   @printf("  % 10.8e", duds_ut)
  #   analysis_callback.save_analysis && @printf(f, "  % 10.8e", duds_ut)
  #   println()
  # end

  # # Entropy
  # if :entropy in analysis_quantities
  #   s = integrate(dg, dg.elements.u) do i, j, element_id, dg, u
  #     cons = get_node_vars(u, dg, i, j, element_id)
  #     return entropy(cons, equations(dg))
  #   end
  #   print(" ∑S:          ")
  #   @printf("  % 10.8e", s)
  #   analysis_callback.save_analysis && @printf(f, "  % 10.8e", s)
  #   println()
  # end

  # TODO: Taal implement additional analysis quantities
  # TODO: Taal refactor, use a tuple of functions that compute the stuff?

  # Add line break and close analysis file if it was opened
  if analysis_callback.save_analysis
    println(io)
    close(io)
  end

  # Return errors for EOC analysis
  return l2_error, linf_error
end


# TODO: Taal refactor, the part above is general, the part below is specialized



function calc_error_norms(func, u::AbstractArray{<:Any,4}, t, analyzer, mesh::TreeMesh{2}, equations, initial_conditions, dg::DGSEM, cache)
  @unpack vandermonde, weights = analyzer
  @unpack node_coordinates = cache.elements

  # pre-allocate buffers
  u_local = zeros(eltype(u),
                  nvariables(equations), nnodes(analyzer), nnodes(analyzer))
  u_tmp1 = similar(u_local,
                   nvariables(equations), nnodes(analyzer), nnodes(dg))
  x_local = zeros(eltype(node_coordinates),
                  ndims(equations), nnodes(analyzer), nnodes(analyzer))
  x_tmp1 = similar(x_local,
                   ndims(equations), nnodes(analyzer), nnodes(dg))

  # Set up data structures
  l2_error   = zero(func(get_node_vars(u, equations, dg, 1, 1, 1), equations))
  linf_error = copy(l2_error)

  # Iterate over all elements for error calculations
  for element in eachelement(dg, cache)
    # Interpolate solution and node locations to analysis nodes
    multiply_dimensionwise!(u_local, vandermonde, view(u,                :, :, :, element), u_tmp1)
    multiply_dimensionwise!(x_local, vandermonde, view(node_coordinates, :, :, :, element), x_tmp1)

    # Calculate errors at each analysis node
    jacobian_volume = inv(cache.elements.inverse_jacobian[element])^ndims(equations)

    for j in eachnode(analyzer), i in eachnode(analyzer)
      u_exact = initial_conditions(get_node_coords(x_local, equations, dg, i, j), t, equations)
      diff = func(u_exact, equations) - func(get_node_vars(u_local, equations, dg, i, j), equations)
      l2_error += diff.^2 * (weights[i] * weights[j] * jacobian_volume)
      linf_error = @. max(linf_error, abs(diff))
    end
  end

  # For L2 error, divide by total volume
  total_volume = mesh.tree.length_level_0^ndims(mesh)
  l2_error = @. sqrt(l2_error / total_volume)

  return l2_error, linf_error
end
