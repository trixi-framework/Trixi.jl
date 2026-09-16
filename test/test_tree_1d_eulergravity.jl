@testsnippet TreeMesh1DEulerGravity begin
    EXAMPLES_DIR = joinpath(examples_dir(), "tree_1d_dgsem")
end

@testitem "TreeMesh1D EulerGravity: elixir_eulergravity_convergence.jl" setup=[
    Setup,
    TreeMesh1DEulerGravity
] tags=[:tree_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_eulergravity_convergence.jl"),
                        l2=[
                            0.00021708496949694728, 0.0002913795242132917,
                            0.0006112500956552259
                        ],
                        linf=[
                            0.0004977733237385706, 0.0013594226727522418,
                            0.0020418739554664
                        ])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
end

@testitem "TreeMesh1D EulerGravity: SaveSolutionCallback saves both systems" setup=[
    Setup,
    TreeMesh1DEulerGravity
] tags=[:tree_part1] begin
    # Start with a clean environment: remove Trixi.jl output directory if it exists
    outdir = "out"
    isdir(outdir) && rm(outdir, recursive = true)

    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_eulergravity_convergence.jl"),
                        tspan=(0.0, 0.1))

    solution_file(system, iter) = joinpath(outdir,
                                           "solution_" * system * "_" *
                                           lpad(iter, 9, '0') * ".h5")

    # Both the initial and the final solution must be saved for both systems
    for iter in (0, sol.stats.naccept)
        @test isfile(solution_file("euler", iter))
        @test isfile(solution_file("gravity", iter))
        # There must not be any file without a system name, which would indicate
        # that the generic (Euler-only) version of `save_solution_file` was used
        @test !isfile(joinpath(outdir, "solution_" * lpad(iter, 9, '0') * ".h5"))
    end

    Trixi.h5open(solution_file("euler", 0), "r") do file
        @test read(Trixi.attributes(file)["equations"]) ==
              "CompressibleEulerEquations1D"
        @test read(Trixi.attributes(file)["n_vars"]) == 3
        @test [read(Trixi.attributes(file["variables_$v"])["name"]) for v in 1:3] ==
              ["rho", "v1", "p"]
    end

    # The gravity file must contain the gravity state stored in `semi.cache.u_ode`
    # and not (a copy of) the Euler state
    u_gravity = Trixi.wrap_array_native(semi.cache.u_ode, semi.semi_gravity)
    Trixi.h5open(solution_file("gravity", sol.stats.naccept), "r") do file
        @test read(Trixi.attributes(file)["equations"]) ==
              "HyperbolicDiffusionEquations1D"
        @test read(Trixi.attributes(file)["n_vars"]) == 2
        @test [read(Trixi.attributes(file["variables_$v"])["name"]) for v in 1:2] ==
              ["phi", "q1"]
        @test read(file["variables_1"]) ≈ vec(u_gravity[1, :, :])
        @test read(file["variables_2"]) ≈ vec(u_gravity[2, :, :])
    end
end
