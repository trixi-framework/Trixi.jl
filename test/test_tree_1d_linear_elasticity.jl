module TestExamples1DLinearElasticity

using Test
using Trixi

include("test_trixi.jl")

EXAMPLES_DIR = joinpath(examples_dir(), "tree_1d_dgsem")

@testset "Linear Elasticity" begin
#! format: noindent

@trixi_testset "elixir_linearelasticity_convergence.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR,
                                 "elixir_linearelasticity_convergence.jl"),
                        analysis_callback=AnalysisCallback(semi,
                                                           interval = analysis_interval,
                                                           extra_analysis_errors = (:l2_error_primitive,
                                                                                    :linf_error_primitive),
                                                           extra_analysis_integrals = (energy_kinetic,
                                                                                       energy_internal,
                                                                                       entropy)),
                        l2=[0.0007205516785218745, 0.0008036755866155103],
                        linf=[0.0011507266875070855, 0.003249818227066381])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)
end

@trixi_testset "elixir_linearelasticity_impact.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_linearelasticity_impact.jl"),
                        l2=[0.004334150310828556, 368790.1916121487],
                        linf=[0.01070558926301203, 999999.9958777003])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)
end
end

end # module
