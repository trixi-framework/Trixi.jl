using Test: @test
import Trixi

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
  local atol = get_kwarg(args, :atol, 500*eps())
  local rtol = get_kwarg(args, :rtol, sqrt(eps()))
  local kwargs = Pair{Symbol, Any}[]
  for arg in args
    if arg.head == :(=) && !(arg.args[1] in (:l2, :linf, :atol, :rtol))
      push!(kwargs, Pair(arg.args...))
    end
  end

  quote
    Trixi.mpi_isroot() && println("═"^100)
    Trixi.mpi_isroot() && println($elixir)

    # evaluate examples in the scope of the module they're called from
    @test_nowarn_mod trixi_include(@__MODULE__, $elixir; $kwargs...)

    # if present, compare l2 and linf errors against reference values
    if !isnothing($l2) || !isnothing($linf)
      l2_measured, linf_measured = analysis_callback(sol)

      if Trixi.mpi_isroot() && !isnothing($l2)
        @test length($l2) == length(l2_measured)
        for (l2_expected, l2_actual) in zip($l2, l2_measured)
          @test isapprox(l2_expected, l2_actual, atol=$atol, rtol=$rtol)
        end
      end

      if Trixi.mpi_isroot() && !isnothing($linf)
        @test length($linf) == length(linf_measured)
        for (linf_expected, linf_actual) in zip($linf, linf_measured)
          @test isapprox(linf_expected, linf_actual, atol=$atol, rtol=$rtol)
        end
      end
    end

    Trixi.mpi_isroot() && println("═"^100)
    Trixi.mpi_isroot() && println("\n\n")
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



# Modified version of `@test_nowarn` that prints the content of `stderr` when
# it is not empty. This is useful for debugging failing tests.
macro test_nowarn_debug(expr)
  quote
    let fname = tempname()
      try
        ret = open(fname, "w") do f
          redirect_stderr(f) do
            $(esc(expr))
          end
        end
        stderr_content = read(fname, String)
        if !isempty(stderr_content)
          println("Content of `stderr`:\n", stderr_content)
        end
        @test isempty(stderr_content)
        ret
      finally
        rm(fname, force=true)
      end
    end
  end
end

# Modified version of `@test_nowarn` that prints the content of `stderr` when
# it is not empty and ignnores module replacements.
macro test_nowarn_mod(expr)
  quote
    let fname = tempname()
      try
        ret = open(fname, "w") do f
          redirect_stderr(f) do
            $(esc(expr))
          end
        end
        stderr_content = read(fname, String)
        if !isempty(stderr_content)
          println("Content of `stderr`:\n", stderr_content)
        end
        # We also ignore simple module redefinitions for convenience. Thus, we
        # check whether every line of `stderr_content` is of the form of a
        # module replacement warning.
        # TODO: Upstream (PlotUtils). This should be removed again once the
        #       deprecated stuff is fixed upstream.
        if stderr_content != "WARNING: importing deprecated binding Colors.RGB1 into PlotUtils.\nWARNING: importing deprecated binding Colors.RGB4 into PlotUtils.\n"
          @test occursin(r"^(WARNING: replacing module .+\.\n)*$", stderr_content)
        end
        ret
      finally
        rm(fname, force=true)
      end
    end
  end
end


"""
    @trixi_testset "name of the testset" #= code to test #=

Similar to `@testset`, but wraps the code inside a temporary module to avoid
namespace pollution. It also `include`s this file again to provide the
definition of `@test_trixi_include`.
"""
macro trixi_testset(name, expr)
  @assert name isa String
  # TODO: `@eval` is evil
  # We would like to use
  #   mod = gensym(name)
  # to create new module names for every test set. However, this is not
  # compatible with the dirty hack using `@eval` to get the mapping when
  # loading structured, curvilinear meshes. Thus, we need to use a plain
  # module name here.
  mod = Symbol("TrixiTestModule")
  quote
    @eval module $mod
      using Test
      using Trixi
      include(@__FILE__)
      # We define `EXAMPLES_DIR` in (nearly) all test modules and use it to
      # get the path to the elixirs to be tested. However, that's not required
      # and we want to fail gracefully if it's not defined.
      try
        import ..EXAMPLES_DIR
      catch
        nothing
      end
      @testset $name $expr
    end
    nothing
  end
end
