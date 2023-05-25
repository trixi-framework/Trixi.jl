# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin


"""
    BoundsCheckCallback(; output_directory="out", save_errors=false)

Bounds checking routine for `IndicatorIDP` and `IndicatorMCL`. Applied as a stage callback for
SSPRK methods.
"""
struct BoundsCheckCallback
  output_directory::String
  save_errors::Bool
end

function BoundsCheckCallback(; output_directory="out", save_errors=false)
  BoundsCheckCallback(output_directory, save_errors)
end

function (callback::BoundsCheckCallback)(u_ode, semi::AbstractSemidiscretization, t, iter, laststage)
  mesh, equations, solver, cache = mesh_equations_solver_cache(semi)
  u = wrap_array(u_ode, mesh, equations, solver, cache)

  @trixi_timeit timer() "check_bounds" check_bounds(u, mesh, equations, solver, cache, t, iter, laststage,
                                                    callback.output_directory, callback.save_errors)
end

function check_bounds(u, mesh, equations, solver, cache, t, iter, laststage, output_directory, save_errors)
  check_bounds(u, mesh, equations, solver, cache, solver.volume_integral, t, iter, laststage, output_directory, save_errors)
end

function check_bounds(u, mesh, equations, solver, cache, volume_integral::AbstractVolumeIntegral, t, iter, laststage, output_directory, save_errors)
  return nothing
end

function check_bounds(u, mesh, equations, solver, cache, volume_integral::VolumeIntegralShockCapturingSubcell,
                      t, iter, laststage, output_directory, save_errors)
  check_bounds(u, mesh, equations, solver, cache, volume_integral.indicator, t, iter,
               laststage, output_directory, save_errors)
end


function init_callback(callback, semi)
  init_callback(callback, semi, semi.solver.volume_integral)
end

init_callback(callback, semi, volume_integral::AbstractVolumeIntegral) = nothing

function init_callback(callback, semi, volume_integral::VolumeIntegralShockCapturingSubcell)
  init_callback(callback, semi, volume_integral.indicator)
end

function init_callback(callback::BoundsCheckCallback, semi, indicator::IndicatorIDP)
  if !callback.save_errors
    return nothing
  end

  @unpack IDPDensityTVD, IDPPressureTVD, IDPPositivity, IDPSpecEntropy, IDPMathEntropy = indicator
  @unpack output_directory = callback
  mkpath(output_directory)
  open("$output_directory/deviations.txt", "a") do f;
    print(f, "# iter, simu_time")
    if IDPDensityTVD
      print(f, ", rho_min, rho_max");
    end
    if IDPPressureTVD
      print(f, ", p_min, p_max");
    end
    if IDPSpecEntropy
      print(f, ", specEntr_min");
    end
    if IDPMathEntropy
      print(f, ", mathEntr_max");
    end
    if IDPPositivity
      for variable in indicator.variables_cons
        if variable == Trixi.density && IDPDensityTVD
          continue
        end
        print(f, ", $(variable)_min");
      end
      for variable in indicator.variables_nonlinear
        if variable == pressure && IDPPressureTVD
          continue
        end
        print(f, ", $(variable)_min");
      end
    end
    println(f)
  end

  return nothing
end

function init_callback(callback::BoundsCheckCallback, semi, indicator::IndicatorMCL)
  if !callback.save_errors
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

function finalize_callback(callback, semi, volume_integral::VolumeIntegralShockCapturingSubcell)
  finalize_callback(callback, semi, volume_integral.indicator)
end


@inline function finalize_callback(callback::BoundsCheckCallback, semi, indicator::IndicatorIDP)
  @unpack IDPDensityTVD, IDPPressureTVD, IDPPositivity, IDPSpecEntropy, IDPMathEntropy = indicator
  @unpack idp_bounds_delta = indicator.cache

  println("─"^100)
  println("Maximum deviation from bounds:")
  println("─"^100)
  counter = 1
  if IDPDensityTVD
    println("rho:\n- lower bound: ", idp_bounds_delta[counter], "\n- upper bound: ", idp_bounds_delta[counter+1])
    counter += 2
  end
  if IDPPressureTVD
    println("pressure:\n- lower bound: ", idp_bounds_delta[counter], "\n- upper bound: ", idp_bounds_delta[counter+1])
    counter += 2
  end
  if IDPSpecEntropy
    println("spec. entropy:\n- lower bound: ", idp_bounds_delta[counter])
    counter += 1
  end
  if IDPMathEntropy
    println("math. entropy:\n- upper bound: ", idp_bounds_delta[counter])
    counter += 1
  end
  if IDPPositivity
    for variable in indicator.variables_cons
      if variable == Trixi.density && IDPDensityTVD
        continue
      end
      println("$(variable):\n- positivity: ", idp_bounds_delta[counter])
      counter += 1
    end
    for variable in indicator.variables_nonlinear
      if variable == pressure && IDPPressureTVD
        continue
      end
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
