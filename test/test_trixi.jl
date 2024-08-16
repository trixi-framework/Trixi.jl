using Test: @test
import Trixi

# Use a macro to avoid world age issues when defining new initial conditions etc.
# inside an elixir.
"""
    @test_trixi_include(elixir; l2=nothing, linf=nothing, RealT=Float64,
                                atol=500*eps(RealT), rtol=sqrt(eps(RealT)),
                                parameters...)

Test Trixi by calling `trixi_include(elixir; parameters...)`.
By default, only the absence of error output is checked.
If `l2` or `linf` are specified, in addition the resulting L2/Linf errors
are compared approximately against these reference values, using `atol, rtol`
as absolute/relative tolerance.
"""
macro test_trixi_include(elixir, args...)
    local l2 = get_kwarg(args, :l2, nothing)
    local linf = get_kwarg(args, :linf, nothing)
    local RealT = get_kwarg(args, :RealT, :Float64)
    if RealT === :Float64
        atol_default = 500 * eps(Float64)
        rtol_default = sqrt(eps(Float64))
    elseif RealT === :Float32
        atol_default = 500 * eps(Float32)
        rtol_default = sqrt(eps(Float32))
    end
    local atol = get_kwarg(args, :atol, atol_default)
    local rtol = get_kwarg(args, :rtol, rtol_default)
    local skip_coverage = get_kwarg(args, :skip_coverage, false)
    local coverage_override = expr_to_named_tuple(get_kwarg(args, :coverage_override, :()))
    if !(:maxiters in keys(coverage_override))
        # maxiters in coverage_override defaults to 1
        coverage_override = (; coverage_override..., maxiters = 1)
    end

    local cmd = string(Base.julia_cmd())
    local coverage = occursin("--code-coverage", cmd) &&
                     !occursin("--code-coverage=none", cmd)

    local kwargs = Pair{Symbol, Any}[]
    for arg in args
        if (arg.head == :(=) &&
            !(arg.args[1] in (:l2, :linf, :RealT, :atol, :rtol, :coverage_override,
                              :skip_coverage))
            && !(coverage && arg.args[1] in keys(coverage_override)))
            push!(kwargs, Pair(arg.args...))
        end
    end

    if coverage
        for key in keys(coverage_override)
            push!(kwargs, Pair(key, coverage_override[key]))
        end
    end

    if coverage && skip_coverage
        return quote
            if Trixi.mpi_isroot()
                println("═"^100)
                println("Skipping coverage test of ", $elixir)
                println("═"^100)
                println("\n\n")
            end
        end
    end

    quote
        Trixi.mpi_isroot() && println("═"^100)
        Trixi.mpi_isroot() && println($elixir)

        # if `maxiters` is set in tests, it is usually set to a small number to
        # run only a few steps - ignore possible warnings coming from that
        if any(==(:maxiters) ∘ first, $kwargs)
            additional_ignore_content = [
                r"┌ Warning: Interrupted\. Larger maxiters is needed\..*\n└ @ SciMLBase .+\n",
                r"┌ Warning: Interrupted\. Larger maxiters is needed\..*\n└ @ Trixi .+\n"]
        else
            additional_ignore_content = []
        end

        # evaluate examples in the scope of the module they're called from
        @test_nowarn_mod trixi_include(@__MODULE__, $elixir; $kwargs...) additional_ignore_content

        # if present, compare l2 and linf errors against reference values
        if !$coverage && (!isnothing($l2) || !isnothing($linf))
            l2_measured, linf_measured = analysis_callback(sol)

            if Trixi.mpi_isroot() && !isnothing($l2)
                @test length($l2) == length(l2_measured)
                for (l2_expected, l2_actual) in zip($l2, l2_measured)
                    @test isapprox(l2_expected, l2_actual, atol = $atol, rtol = $rtol)
                end
            end

            if Trixi.mpi_isroot() && !isnothing($linf)
                @test length($linf) == length(linf_measured)
                for (linf_expected, linf_actual) in zip($linf, linf_measured)
                    @test isapprox(linf_expected, linf_actual, atol = $atol, rtol = $rtol)
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

function expr_to_named_tuple(expr)
    result = (;)

    for arg in expr.args
        if arg.head != :(=)
            error("Invalid expression")
        end
        result = (; result..., arg.args[1] => arg.args[2])
    end
    return result
end

# Modified version of `@test_nowarn` that prints the content of `stderr` when
# it is not empty and ignores module replacements.
macro test_nowarn_mod(expr, additional_ignore_content = String[])
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

                # Patterns matching the following ones will be ignored. Additional patterns
                # passed as arguments can also be regular expressions, so we just use the
                # type `Any` for `ignore_content`.
                ignore_content = Any[
                                     # We need to ignore steady state information reported by our callbacks
                                     r"┌ Info:   Steady state tolerance reached\n│   steady_state_callback .+\n└   t = .+\n",
                                     # We also ignore our own compilation messages
                                     "[ Info: You just called `trixi_include`. Julia may now compile the code, please be patient.\n",
                                     # TODO: Upstream (PlotUtils). This should be removed again once the
                                     #       deprecated stuff is fixed upstream.
                                     "WARNING: importing deprecated binding Colors.RGB1 into Plots.\n",
                                     "WARNING: importing deprecated binding Colors.RGB4 into Plots.\n",
                                     r"┌ Warning: Keyword argument letter not supported with Plots.+\n└ @ Plots.+\n",
                                     r"┌ Warning: `parse\(::Type, ::Coloarant\)` is deprecated.+\n│.+\n│.+\n└ @ Plots.+\n",
                                     # TODO: Silence warning introduced by Flux v0.13.13. Should be properly fixed.
                                     r"┌ Warning: Layer with Float32 parameters got Float64 input.+\n│.+\n│.+\n│.+\n└ @ Flux.+\n",
                                     # NOTE: These warnings arose from Julia 1.10 onwards
                                     r"WARNING: Method definition .* in module .* at .* overwritten .*.\n",
                                     # Warnings from third party packages
                                     r"┌ Warning: Problem status ALMOST_INFEASIBLE; solution may be inaccurate.\n└ @ Convex ~/.julia/packages/Convex/.*\n",
                                     r"┌ Warning: Problem status ALMOST_OPTIMAL; solution may be inaccurate.\n└ @ Convex ~/.julia/packages/Convex/.*\n"]
                append!(ignore_content, $additional_ignore_content)
                for pattern in ignore_content
                    stderr_content = replace(stderr_content, pattern => "")
                end

                # We also ignore simple module redefinitions for convenience. Thus, we
                # check whether every line of `stderr_content` is of the form of a
                # module replacement warning.
                @test occursin(r"^(WARNING: replacing module .+\.\n)*$", stderr_content)
                ret
            finally
                rm(fname, force = true)
            end
        end
    end
end

"""
    @timed_testset "name of the testset" #= code to test #=

Similar to `@testset`, but prints the name of the testset and its runtime
after execution.
"""
macro timed_testset(name, expr)
    @assert name isa String
    quote
        local time_start = time_ns()
        @testset $name $expr
        local time_stop = time_ns()
        if Trixi.mpi_isroot()
            flush(stdout)
            @info("Testset "*$name*" finished in "
                  *string(1.0e-9 * (time_stop - time_start))*" seconds.\n")
            flush(stdout)
        end
    end
end

"""
    @trixi_testset "name of the testset" #= code to test #=

Similar to `@testset`, but wraps the code inside a temporary module to avoid
namespace pollution. It also `include`s this file again to provide the
definition of `@test_trixi_include`. Moreover, it records the execution time
of the testset similarly to [`timed_testset`](@ref).
"""
macro trixi_testset(name, expr)
    @assert name isa String
    # TODO: `@eval` is evil
    # We would like to use
    #   mod = gensym(name)
    #   ...
    #   module $mod
    # to create new module names for every test set. However, this is not
    # compatible with the dirty hack using `@eval` to get the mapping when
    # loading structured, curvilinear meshes. Thus, we need to use a plain
    # module name here.
    quote
        local time_start = time_ns()
        @eval module TrixiTestModule
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
        local time_stop = time_ns()
        if Trixi.mpi_isroot()
            flush(stdout)
            @info("Testset "*$name*" finished in "
                  *string(1.0e-9 * (time_stop - time_start))*" seconds.\n")
        end
        nothing
    end
end
