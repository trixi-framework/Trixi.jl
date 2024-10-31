using Test
using Trixi

include("../test/test_trixi.jl")

EXAMPLES_DIR = joinpath(examples_dir(), "p4est_3d_dgsem")

@trixi_testset "elixir_mhd_alfven_wave_nonconforming.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR,
                                 "elixir_mhd_alfven_wave_nonconforming.jl"),
                        l2=[
                            0.0019000780137625318,
                            0.006850450489861558,
                            0.003417425519997495,
                            0.009175507419487644,
                            0.007684415188358489,
                            0.008202320974386567,
                            0.003746708238600488,
                            0.009259762088450122,
                            1.9204077712342107e-5
                        ],
                        linf=[
                            0.014386878844156015,
                            0.040741566943627766,
                            0.020731713202942308,
                            0.055715997178527876,
                            0.04852444163354308,
                            0.047070868014941314,
                            0.030255581142574206,
                            0.06164784183754593,
                            0.00022501258622770076
                        ],
                        tspan=(0.0, 0.25), trees_per_dimension=(3, 1, 1),
                        coverage_override=(trees_per_dimension = (1, 1, 1),))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    # let
    #     t = sol.t[end]
    #     u_ode = sol.u[end]
    #     du_ode = similar(u_ode)
    #     @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    # end
end
