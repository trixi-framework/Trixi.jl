using Test: @test_nowarn, @test
import Trixi


"""
    test_trixi_run(parameters_file; l2=nothing, linf=nothing, atol=10*eps(), rtol=0.001, parameters...)

Test Trixi by calling `Trixi.run(parameters_file; parameters...)`.
By default, only the absence of error output is checked.
If `l2` or `linf` are specified, in addition the resulting L2/Linf errors
are compared approximately against these reference values, using `atol, rtol`
as absolute/relative tolerance.
"""
function test_trixi_run(parameters_file; l2=nothing, linf=nothing, atol=200*eps(), rtol=0.001, parameters...)
  # Run basic test to ensure that there is no output to STDERR
  l2_measured, linf_measured, _ = @test_nowarn Trixi.run(parameters_file; parameters...)

  # If present, compare L2 and Linf errors against reference values
  if !isnothing(l2)
    for (l2_expected, l2_actual) in zip(l2, l2_measured)
      @test isapprox(l2_expected, l2_actual, atol=atol, rtol=rtol)
    end
  end
  if !isnothing(linf)
    for (linf_expected, linf_actual) in zip(linf, linf_measured)
      @test isapprox(linf_expected, linf_actual, atol=atol, rtol=rtol)
    end
  end
end


# Get the first value assigned to `keyword` in `args` and return `default_value`
# if there are no assignments to `keyword` in `args`.
function get_kwarg(args, keyword, default_value)
  val = default_value
  for arg in args
    if arg.head == :(=) && arg.args[1] == keyword
      val = arg.args[2]
      break
    end
  end
  return val
end

# Use a macro to avoid world age issues when defining new initial conditions etc.
# inside an elixir.
"""
    @test_trixi_include(elixir; l2=nothing, linf=nothing,
                                atol=10*eps(), rtol=0.001,
                                parameters...)

Test Trixi by calling `trixi_include(elixir; parameters...)`.
By default, only the absence of error output is checked.
If `l2` or `linf` are specified, in addition the resulting L2/Linf errors
are compared approximately against these reference values, using `atol, rtol`
as absolute/relative tolerance.
"""
macro test_trixi_include(elixir, args...)

  local l2   = get_kwarg(args, :l2, nothing)
  local linf = get_kwarg(args, :linf, nothing)
  local atol = get_kwarg(args, :atol, 200*eps())
  local rtol = get_kwarg(args, :rtol, 0.001)
  local kwargs = Pair{Symbol, Any}[]
  for arg in args
    if arg.head == :(=) && !(arg.args[1] in (:l2, :linf, :atol, :rtol))
      push!(kwargs, Pair(arg.args...))
    end
  end

  quote
    println("#"^80)
    println($elixir)

    # evaluate examples in the scope of the module they're called from
    @test_nowarn trixi_include(@__MODULE__, $elixir; $kwargs...)

    # if present, compare l2 and linf errors against reference values
    if !isnothing($l2) || !isnothing($linf)
      l2_measured, linf_measured = analysis_callback(sol)

      if !isnothing($l2)
        for (l2_expected, l2_actual) in zip($l2, l2_measured)
          @test isapprox(l2_expected, l2_actual, atol=$atol, rtol=$rtol)
        end
      end

      if !isnothing($linf)
        for (linf_expected, linf_actual) in zip($linf, linf_measured)
          @test isapprox(linf_expected, linf_actual, atol=$atol, rtol=$rtol)
        end
      end
    end

    println("#"^80)
    println("\n\n")
  end
end
