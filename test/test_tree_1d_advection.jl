@testsnippet TreeMesh1DAdvection begin
    EXAMPLES_DIR = joinpath(examples_dir(), "tree_1d_dgsem")
end

@testitem "elixir_advection_basic.jl" setup=[Setup, TreeMesh1DAdvection] tags=[:tree_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_basic.jl"),
                        l2=[6.0388296447998465e-6],
                        linf=[3.217887726258972e-5])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)
end

@testitem "elixir_advection_gauss_legendre.jl" setup=[Setup, TreeMesh1DAdvection] tags=[:tree_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_gauss_legendre.jl"),
                        l2=[2.515203865524688e-6], linf=[8.660338936650191e-6])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)
end

@testitem "elixir_advection_limiter_liu_zhang.jl" setup=[Setup, TreeMesh1DAdvection] tags=[:tree_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR,
                                 "elixir_advection_limiter_liu_zhang.jl"),
                        l2=[0.09842318275842536],
                        linf=[0.5084209598077918])
    u = Trixi.wrap_array_native(sol.u[end], semi)
    # matches thresholds = (1e-3,) up to a tolerance
    @test minimum(u) > 1e-3 - 10 * eps()
    @test_allocations(Trixi.rhs!, semi, sol, 1000)
end

@testitem "elixir_advection_basic.jl (max_abs_speed)" setup=[Setup, TreeMesh1DAdvection] tags=[:tree_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR,
                                 "elixir_advection_limiter_liu_zhang_amr.jl"),
                        l2=[0.08104042028981012], linf=[0.5248014378268002])
    u = Trixi.wrap_array_native(sol.u[end], semi)
    # matches thresholds = (1e-3,) up to a tolerance
    @test minimum(u) > 1e-3 - 10 * eps()
    @test_allocations(Trixi.rhs!, semi, sol, 1000)
end

@testitem "elixir_advection_basic.jl (max_abs_speed)" setup=[Setup, TreeMesh1DAdvection] tags=[:tree_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_basic.jl"),
                        surface_flux=FluxLaxFriedrichs(max_abs_speed),
                        l2=[6.0388296447998465e-6],
                        linf=[3.217887726258972e-5])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)
end

@testitem "elixir_advection_amr.jl" setup=[Setup, TreeMesh1DAdvection] tags=[:tree_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_amr.jl"),
                        l2=[0.3540206249507417],
                        linf=[0.9999896603382347])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)
end

@testitem "elixir_advection_amr_nonperiodic.jl" setup=[Setup, TreeMesh1DAdvection] tags=[:tree_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_amr_nonperiodic.jl"),
                        l2=[4.283508859843524e-6],
                        linf=[3.235356127918171e-5])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)
end

@testitem "elixir_advection_basic.jl (No errors)" setup=[Setup, TreeMesh1DAdvection] tags=[:tree_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_basic.jl"),
                        analysis_callback=AnalysisCallback(semi, interval = 42,
                                                           analysis_errors = Symbol[]))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)
end

@testitem "elixir_advection_convergence_fvO2.jl" setup=[Setup, TreeMesh1DAdvection] tags=[:tree_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_convergence_fvO2.jl"),
                        l2=[0.0024544920169555706], linf=[0.007837347144210138])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)
end

@testitem "elixir_advection_finite_volume.jl" setup=[Setup, TreeMesh1DAdvection] tags=[:tree_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_finite_volume.jl"),
                        l2=[0.011662300515980219],
                        linf=[0.01647256923710194])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)
end

@testitem "elixir_advection_perk2.jl" setup=[Setup, TreeMesh1DAdvection] tags=[:tree_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_perk2.jl"),
                        l2=[0.011288030389423475],
                        linf=[0.01596735472556976])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    # Larger values for allowed allocations due to usage of custom
    # integrator which are not *recorded* for the methods from
    # OrdinaryDiffEq.jl
    # Corresponding issue: https://github.com/trixi-framework/Trixi.jl/issues/1877
    @test_allocations(Trixi.rhs!, semi, sol, 8000)
end

# Testing the second-order paired explicit Runge-Kutta (PERK) method without stepsize callback
@testitem "elixir_advection_perk2.jl(fixed time step)" setup=[Setup, TreeMesh1DAdvection] tags=[:tree_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_perk2.jl"),
                        dt=2.0e-3,
                        tspan=(0.0, 20.0),
                        save_solution=SaveSolutionCallback(dt = 0.1 + 1.0e-8),
                        callbacks=CallbackSet(summary_callback, save_solution,
                                              analysis_callback, alive_callback),
                        l2=[9.886271430207691e-6],
                        linf=[3.729460413781638e-5])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    # Larger values for allowed allocations due to usage of custom
    # integrator which are not *recorded* for the methods from
    # OrdinaryDiffEq.jl
    # Corresponding issue: https://github.com/trixi-framework/Trixi.jl/issues/1877
    @test_allocations(Trixi.rhs!, semi, sol, 8000)
end

# Testing the second-order paired explicit Runge-Kutta (PERK) method with the optimal CFL number
@testitem "elixir_advection_perk2_optimal_cfl.jl" setup=[Setup, TreeMesh1DAdvection] tags=[:tree_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_perk2_optimal_cfl.jl"),
                        l2=[0.0009700887119146429],
                        linf=[0.00137209242077041])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    # Larger values for allowed allocations due to usage of custom
    # integrator which are not *recorded* for the methods from
    # OrdinaryDiffEq.jl
    # Corresponding issue: https://github.com/trixi-framework/Trixi.jl/issues/1877
    @test_allocations(Trixi.rhs!, semi, sol, 8000)
end

# TODO (TestItems.jl migration): This test uses typed reference values
# (`l2 = Double64[...]`). `TrixiTest`'s `@test_trixi_include_base` splices the
# `l2`/`linf` reference values into its comparison loop *unescaped*, so the bare
# `Double64` is resolved in the `TrixiTest` module instead of this test item and
# errors with `UndefVarError: Double64 not defined in TrixiTest`. The legacy
# `@trixi_testset` masked this because it imported `Double64` into the same
# temporary module that evaluated the comparison. Re-enable once `TrixiTest`
# escapes those reference values (or another fix is agreed upon).
# @testitem "elixir_advection_doublefloat.jl" setup=[Setup, TreeMesh1DAdvection] tags=[:tree_part1] begin
#     using DoubleFloats: Double64
#     @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_doublefloat.jl"),
#                         l2=Double64[6.80895929885700039832943251427357703e-11],
#                         linf=Double64[5.82834770064525291688100323411704252e-10])
#     # Ensure that we do not have excessive memory allocations
#     # (e.g., from type instabilities)
#     @test_allocations(Trixi.rhs!, semi, sol, 1000)
# end
