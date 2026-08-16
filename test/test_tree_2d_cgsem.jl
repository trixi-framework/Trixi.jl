@testsnippet TreeMesh2DCGSEM begin
    EXAMPLES_DIR = joinpath(examples_dir(), "tree_2d_cgsem")
end

@testitem "TreeMesh2D CGSEM: elixir_advection_basic.jl" setup=[
    Setup,
    TreeMesh2DCGSEM
] tags=[:tree_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_basic.jl"),
                        l2=[8.752975685966706e-6],
                        linf=[4.4382874226034374e-5])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
end

@testitem "TreeMesh2D CGSEM: elixir_euler_ec.jl" setup=[
    Setup,
    TreeMesh2DCGSEM
] tags=[:tree_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_ec.jl"),
                        l2=[
                            0.061777911126567026,
                            0.05012708271374828,
                            0.050129362317294264,
                            0.22599176450467243
                        ],
                        linf=[
                            0.30542503728734316,
                            0.3021446902295008,
                            0.29716183278197195,
                            1.0957646414396867
                        ])
    # The CGSEM is entropy conservative, i.e., the semidiscrete rate of change of
    # the total entropy vanishes up to round-off errors
    du_ode = similar(sol.u[end])
    Trixi.rhs_hyperbolic!(du_ode, sol.u[end], semi, sol.t[end])
    dsdt = Trixi.analyze(Trixi.entropy_timederivative,
                         Trixi.wrap_array(du_ode, semi),
                         Trixi.wrap_array(sol.u[end], semi), sol.t[end],
                         Trixi.mesh_equations_solver_cache(semi)...)
    @test abs(dsdt) < 1.0e-12

    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
end

@testitem "TreeMesh2D CGSEM: elixir_euler_convergence.jl" setup=[
    Setup,
    TreeMesh2DCGSEM
] tags=[:tree_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_convergence.jl"),
                        l2=[
                            0.00418069878447206,
                            0.0032874130972270514,
                            0.003287413097227444,
                            0.004101965746955797
                        ],
                        linf=[
                            0.01282314230503978,
                            0.012618710275249523,
                            0.0126187102752513,
                            0.014608699953676307
                        ])

    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
end
