using Test: @test
using TrixiTest

macro test_trixi_include(expr, args...)
    local add_to_additional_ignore_content = [
        # We need to ignore steady state information reported by our callbacks
        r"┌ Info:   Steady state tolerance reached\n│   steady_state_callback .+\n└   t = .+\n",
        # NOTE: These warnings arose from Julia 1.10 onwards
        r"WARNING: Method definition .* in module .* at .* overwritten .*.\n",
        # Warnings from third party packages
        r"┌ Warning: Problem status ALMOST_INFEASIBLE; solution may be inaccurate.\n└ @ Convex ~/.julia/packages/Convex/.*\n",
        r"┌ Warning: Problem status ALMOST_OPTIMAL; solution may be inaccurate.\n└ @ Convex ~/.julia/packages/Convex/.*\n",
        # Warnings for higher-precision floating data types
        r"┌ Warning: #= /home/runner/work/Trixi.jl/Trixi.jl/src/solvers/dgsem/interpolation.jl:118 =#:\n│ `LoopVectorization.check_args` on your inputs failed; running fallback `@inbounds @fastmath` loop instead.\n│ Use `warn_check_args=false`, e.g. `@turbo warn_check_args=false ...`, to disable this warning.\n└ @ Trixi ~/.julia/packages/LoopVectorization/.*\n",
        r"┌ Warning: #= /home/runner/work/Trixi.jl/Trixi.jl/src/solvers/dgsem/interpolation.jl:136 =#:\n│ `LoopVectorization.check_args` on your inputs failed; running fallback `@inbounds @fastmath` loop instead.\n│ Use `warn_check_args=false`, e.g. `@turbo warn_check_args=false ...`, to disable this warning.\n└ @ Trixi ~/.julia/packages/LoopVectorization/.*\n"
    ]
    args = append_to_kwargs(args, :additional_ignore_content,
                            add_to_additional_ignore_content)
    quote
        @test_trixi_include_base($(esc(expr)), $(args...))
    end
end
