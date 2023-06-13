# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin


"""
    BoundsCheckCallback(; output_directory="out", save_errors=false, interval=1)

Bounds checking routine for `IndicatorIDP` and `IndicatorMCL`. Applied as a stage callback for
SSPRK methods. If `save_errors` is `true`, the resulting deviations are saved in
`output_directory/deviations.txt` for every `interval` time steps.
"""
struct BoundsCheckCallback
  output_directory::String
  save_errors::Bool
  interval::Int
end

function BoundsCheckCallback(; output_directory="out", save_errors=false, interval=1)
  BoundsCheckCallback(output_directory, save_errors, interval)
end

function (callback::BoundsCheckCallback)(u_ode, integrator, stage)
  mesh, equations, solver, cache = mesh_equations_solver_cache(integrator.p)
  @unpack t, iter, alg = integrator
  u = wrap_array(u_ode, mesh, equations, solver, cache)

  save_errors_ = callback.save_errors && (callback.interval > 0) && (stage == length(alg.c))
  @trixi_timeit timer() "check_bounds" check_bounds(u, mesh, equations, solver, cache, t, iter+1,
      callback.output_directory, save_errors_, callback.interval)
end

function check_bounds(u, mesh, equations, solver, cache, t, iter, output_directory, save_errors, interval)
  check_bounds(u, mesh, equations, solver, cache, solver.volume_integral, t, iter,
               output_directory, save_errors, interval)
end

function check_bounds(u, mesh, equations, solver, cache, volume_integral::AbstractVolumeIntegral,
                      t, iter, output_directory, save_errors, interval)
  return nothing
end

function check_bounds(u, mesh, equations, solver, cache, volume_integral::VolumeIntegralSubcellLimiting,
                      t, iter, output_directory, save_errors, interval)
  check_bounds(u, mesh, equations, solver, cache, volume_integral.indicator, t, iter,
               output_directory, save_errors, interval)
end


function init_callback(callback, semi)
  init_callback(callback, semi, semi.solver.volume_integral)
end

init_callback(callback, semi, volume_integral::AbstractVolumeIntegral) = nothing

function init_callback(callback, semi, volume_integral::VolumeIntegralSubcellLimiting)
  init_callback(callback, semi, volume_integral.indicator)
end

function init_callback(callback::BoundsCheckCallback, semi, indicator::IndicatorIDP)
  if !callback.save_errors || (callback.interval == 0)
    return nothing
  end

  @unpack state_tvd, positivity, spec_entropy, math_entropy = indicator
  @unpack output_directory = callback
  variables = varnames(cons2cons, semi.equations)

  mkpath(output_directory)
  open("$output_directory/deviations.txt", "a") do f;
    print(f, "# iter, simu_time")
    if state_tvd
      for index in indicator.variables_states
        print(f, ", $(variables[index])_min, $(variables[index])_max");
      end
    end
    if spec_entropy
      print(f, ", specEntr_min");
    end
    if math_entropy
      print(f, ", mathEntr_max");
    end
    if positivity
      for index in indicator.variables_cons
        if state_tvd && index in indicator.variables_states
          continue
        end
        print(f, ", $(variables[index])_min");
      end
      for variable in indicator.variables_nonlinear
        print(f, ", $(variable)_min");
      end
    end
    println(f)
  end

  return nothing
end

function init_callback(callback::BoundsCheckCallback, semi, indicator::IndicatorMCL)
  if !callback.save_errors || (callback.interval == 0)
    return nothing
  end

  @unpack output_directory = callback
  mkpath(output_directory)
  open("$output_directory/deviations.txt", "a") do f;
    print(f, "# iter, simu_time", join(", $(v)_min, $(v)_max" for v in varnames(cons2cons, semi.equations)));
    if indicator.PressurePositivityLimiterKuzmin
      print(f, ", pressure_min")
    end
    # No check for entropy limiting rn
    println(f)
  end

  return nothing
end


function finalize_callback(callback, semi)
  finalize_callback(callback, semi, semi.solver.volume_integral)
end

finalize_callback(callback, semi, volume_integral::AbstractVolumeIntegral) = nothing

function finalize_callback(callback, semi, volume_integral::VolumeIntegralSubcellLimiting)
  finalize_callback(callback, semi, volume_integral.indicator)
end


@inline function finalize_callback(callback::BoundsCheckCallback, semi, indicator::IndicatorIDP)
  @unpack state_tvd, positivity, spec_entropy, math_entropy = indicator
  @unpack idp_bounds_delta = indicator.cache
  variables = varnames(cons2cons, semi.equations)

  println("─"^100)
  println("Maximum deviation from bounds:")
  println("─"^100)
  counter = 1
  if state_tvd
    for index in indicator.variables_states
      println("$(variables[index]):")
      println("-lower bound: ", idp_bounds_delta[counter])
      println("-upper bound: ", idp_bounds_delta[counter + 1])
      counter += 2
    end
  end
  if spec_entropy
    println("spec. entropy:\n- lower bound: ", idp_bounds_delta[counter])
    counter += 1
  end
  if math_entropy
    println("math. entropy:\n- upper bound: ", idp_bounds_delta[counter])
    counter += 1
  end
  if positivity
    for index in indicator.variables_cons
      if state_tvd && (index in indicator.variables_states)
        continue
      end
      println("$(variables[index]):\n- positivity: ", idp_bounds_delta[counter])
      counter += 1
    end
    for variable in indicator.variables_nonlinear
      println("$(variable):\n- positivity: ", idp_bounds_delta[counter])
      counter += 1
    end
  end
  println("─"^100 * "\n")

  return nothing
end


@inline function finalize_callback(callback::BoundsCheckCallback, semi, indicator::IndicatorMCL)
  @unpack idp_bounds_delta = indicator.cache

  println("─"^100)
  println("Maximum deviation from bounds:")
  println("─"^100)
  variables = varnames(cons2cons, semi.equations)
  for v in eachvariable(semi.equations)
    println(variables[v], ":\n- lower bound: ", idp_bounds_delta[1, v], "\n- upper bound: ", idp_bounds_delta[2, v])
  end
  if indicator.PressurePositivityLimiterKuzmin
    println("pressure:\n- lower bound: ", idp_bounds_delta[1, nvariables(semi.equations)+1])
  end
  println("─"^100 * "\n")

  return nothing
end


include("bounds_check_2d.jl")


end # @muladd
