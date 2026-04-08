using Test: @test, @testset
using TrixiTest
using Trixi: examples_dir

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
        r"┌ Warning: #= /home/runner/work/Trixi.jl/Trixi.jl/src/solvers/dgsem/interpolation.jl:136 =#:\n│ `LoopVectorization.check_args` on your inputs failed; running fallback `@inbounds @fastmath` loop instead.\n│ Use `warn_check_args=false`, e.g. `@turbo warn_check_args=false ...`, to disable this warning.\n└ @ Trixi ~/.julia/packages/LoopVectorization/.*\n",
        # Warnings for upstream problems in OrdinaryDiffEqSDIRK.jl/OrdinaryDiffEqDifferentiation.jl
        r"┌ Warning: The call to compilecache failed to create a usable precompiled cache file for OrdinaryDiffEqSDIRK \[2d112036-d095-4a1e-ab9a-08536f3ecdbf\]\n│   exception = Required dependency OrdinaryDiffEqDifferentiation \[4302a76b-040a-498a-8c04-15b101fed76b\] failed to load from a cache file.\n└ @ Base loading.jl:.+\n"
    ]
    # if `maxiters` is set in tests, it is usually set to a small number to
    # run only a few steps - ignore possible warnings coming from that
    if any(expr.args[1] == (:maxiters) for expr in args)
        push!(add_to_additional_ignore_content,
              r"┌ Warning: Verbosity toggle: max_iters \n│  Interrupted\. Larger maxiters is needed\..*\n└ @ Trixi .+\n",
              r"┌ Warning: Interrupted\. Larger maxiters is needed\..*\n└ @ Trixi .+\n")
    end
    args = append_to_kwargs(args, :additional_ignore_content,
                            add_to_additional_ignore_content)
    ex = quote
        @test_trixi_include_base($expr, $(args...))
    end
    return esc(ex)
end
