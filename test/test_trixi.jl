using Test: @test
using TrixiTest: @trixi_test_nowarn
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
    # Note: The variables below are just Symbols, not actual errors/types
    local l2 = get_kwarg(args, :l2, nothing)
    local linf = get_kwarg(args, :linf, nothing)
    local RealT = get_kwarg(args, :RealT, :Float64)
    if RealT === :Float64
        atol_default = 500 * eps(Float64)
        rtol_default = sqrt(eps(Float64))
    elseif RealT === :Float32
        atol_default = 500 * eps(Float32)
        rtol_default = sqrt(eps(Float32))
    elseif RealT === :Float128
        atol_default = 500 * eps(Float128)
        rtol_default = sqrt(eps(Float128))
    elseif RealT === :Double64
        atol_default = 500 * eps(Double64)
        rtol_default = sqrt(eps(Double64))
    end
    local atol = get_kwarg(args, :atol, atol_default)
    local rtol = get_kwarg(args, :rtol, rtol_default)

    local kwargs = Pair{Symbol, Any}[]
    for arg in args
        if (arg.head == :(=) &&
            !(arg.args[1] in (:l2, :linf, :RealT, :atol, :rtol)))
            push!(kwargs, Pair(arg.args...))
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
        if !isnothing($l2) || !isnothing($linf)
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

macro test_nowarn_mod(expr, additional_ignore_content = [])
    quote
        add_to_additional_ignore_content = [
            # We need to ignore steady state information reported by our callbacks
            r"┌ Info:   Steady state tolerance reached\n│   steady_state_callback .+\n└   t = .+\n",
            # NOTE: These warnings arose from Julia 1.10 onwards
            r"WARNING: Method definition .* in module .* at .* overwritten at .*",
            # Warnings from third party packages
            r"┌ Warning: Problem status ALMOST_INFEASIBLE; solution may be inaccurate.\n└ @ Convex ~/.julia/packages/Convex/.*\n",
            r"┌ Warning: Problem status ALMOST_OPTIMAL; solution may be inaccurate.\n└ @ Convex ~/.julia/packages/Convex/.*\n",
            # Warnings for higher-precision floating data types or Measurements.jl
            r"┌ Warning: #= /home/runner/work/Trixi.jl/Trixi.jl/src/solvers/dgsem/interpolation.jl:118 =#:\n│ `LoopVectorization.check_args` on your inputs failed; running fallback `@inbounds @fastmath` loop instead.\n│ Use `warn_check_args=false`, e.g. `@turbo warn_check_args=false ...`, to disable this warning.\n└ @ Trixi ~/.julia/packages/LoopVectorization/.*\n",
            r"┌ Warning: #= /home/runner/work/Trixi.jl/Trixi.jl/src/solvers/dgsem/interpolation.jl:136 =#:\n│ `LoopVectorization.check_args` on your inputs failed; running fallback `@inbounds @fastmath` loop instead.\n│ Use `warn_check_args=false`, e.g. `@turbo warn_check_args=false ...`, to disable this warning.\n└ @ Trixi ~/.julia/packages/LoopVectorization/.*\n",
            # Warnings for abstract Real floating point types
            r"┌ Warning: #= /home/runner/work/Trixi.jl/Trixi.jl/src/solvers/dgsem_structured/containers_2d.jl:73 =#:\n│ `LoopVectorization.check_args` on your inputs failed; running fallback `@inbounds @fastmath` loop instead.\n│ Use `warn_check_args=false`, e.g. `@turbo warn_check_args=false ...`, to disable this warning.\n└ @ Trixi ~/.julia/packages/LoopVectorization/.*\n",
            r"┌ Warning: #= /home/runner/work/Trixi.jl/Trixi.jl/src/solvers/dgsem_structured/containers_2d.jl:87 =#:\n│ `LoopVectorization.check_args` on your inputs failed; running fallback `@inbounds @fastmath` loop instead.\n│ Use `warn_check_args=false`, e.g. `@turbo warn_check_args=false ...`, to disable this warning.\n└ @ Trixi ~/.julia/packages/LoopVectorization/.*\n"
        ]
        append!($additional_ignore_content, add_to_additional_ignore_content)
        @trixi_test_nowarn $(esc(expr)) $additional_ignore_content
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
