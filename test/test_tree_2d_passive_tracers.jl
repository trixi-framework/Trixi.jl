@testsnippet TreeMesh2DPassiveTracers begin
    EXAMPLES_DIR = joinpath(examples_dir(), "tree_2d_dgsem")
end

@testitem "TreeMesh2D Passive Tracers: elixir_euler_density_wave_tracers.jl" setup=[
    Setup,
    TreeMesh2DPassiveTracers
] tags=[:tree_part2] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_density_wave_tracers.jl"),
                        l2=[
                            0.0012704690524147188,
                            0.00012704690527390463,
                            0.00025409381047976197,
                            3.17617263147723e-5,
                            0.0527467468452892,
                            0.052788143280791185
                        ],
                        linf=[
                            0.0071511674295154926,
                            0.0007151167435655859,
                            0.0014302334865533006,
                            0.00017877918656949987,
                            0.2247919517756231,
                            0.2779841048041337
                        ])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
end

@testitem "TreeMesh2D Passive Tracers: elixir_euler_density_wave_tracers_es.jl" setup=[
    Setup,
    TreeMesh2DPassiveTracers
] tags=[:tree_part2] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_density_wave_tracers_es.jl"),
                        l2=[
                            0.025062592281985988,
                            0.002506259228198258,
                            0.005012518456396704,
                            0.0006265648070465291,
                            0.030285698454919602,
                            0.030308933771940296
                        ],
                        linf=[
                            0.14150628992886105,
                            0.014150628992890352,
                            0.028301257985774486,
                            0.0035376572482732627,
                            0.23960450439079706,
                            0.13674885136098042
                        ])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)

    # The total entropy decreases for the entropy-stable discretization
    @test Trixi.integrate(entropy, sol.u[end], semi) <
          Trixi.integrate(entropy, sol.u[1], semi)
end
