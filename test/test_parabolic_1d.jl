@testsnippet Parabolic1D begin
    EXAMPLES_DIR = examples_dir()
end

@testitem "Parabolic1D: TreeMesh1D: elixir_advection_diffusion.jl" setup=[
    Setup,
    Parabolic1D
] tags=[:parabolic_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "tree_1d_dgsem",
                                 "elixir_advection_diffusion.jl"),
                        initial_refinement_level=4, tspan=(0.0, 0.4), polydeg=3,
                        l2=[8.40483031802723e-6],
                        linf=[2.8990878868540015e-5])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
end

@testitem "Parabolic1D: TreeMesh1D: elixir_advection_diffusion_ldg.jl" setup=[
    Setup,
    Parabolic1D
] tags=[:parabolic_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "tree_1d_dgsem",
                                 "elixir_advection_diffusion_ldg.jl"),
                        initial_refinement_level=4, tspan=(0.0, 0.4), polydeg=3,
                        l2=[9.234438322146518e-6], linf=[5.425491770139068e-5])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
end

@testitem "Parabolic1D: TreeMesh1D: elixir_advection_diffusion_ldg.jl (Gauss-Legendre)" setup=[
    Setup,
    Parabolic1D
] tags=[:parabolic_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "tree_1d_dgsem",
                                 "elixir_advection_diffusion_ldg.jl"),
                        solver=DGSEM(polydeg = 3, surface_flux = flux_lax_friedrichs,
                                     basis_type = GaussLegendreBasis),
                        tspan=(0.0, 0.4),
                        l2=[4.126471023759558e-6], linf=[1.4470099431229677e-5])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
end

@testitem "Parabolic1D: TreeMesh1D: elixir_advection_diffusion_gradient_source_terms.jl" setup=[
    Setup,
    Parabolic1D
] tags=[:parabolic_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "tree_1d_dgsem",
                                 "elixir_advection_diffusion_gradient_source_terms.jl"),
                        initial_refinement_level=4, tspan=(0.0, 0.4), polydeg=3,
                        l2=[1.0990454698899562e-5], linf=[6.469747978055107e-5])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
end

@testitem "Parabolic1D: TreeMesh1D: elixir_advection_diffusion_restart.jl" setup=[
    Setup,
    Parabolic1D
] tags=[:parabolic_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "tree_1d_dgsem",
                                 "elixir_advection_diffusion_restart.jl"),
                        l2=[1.0679933947301556e-5],
                        linf=[3.910500545667439e-5])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
end

@testitem "Parabolic1D: TreeMesh1D: elixir_advection_diffusion_cfl.jl" setup=[
    Setup,
    Parabolic1D
] tags=[:parabolic_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "tree_1d_dgsem",
                                 "elixir_advection_diffusion_cfl.jl"),
                        l2=[6.763177530985864e-5], linf=[0.0002344578097126515])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
end

@testitem "Parabolic1D: TreeMesh1D: elixir_advection_diffusion_dirichlet_amr.jl" setup=[
    Setup,
    Parabolic1D
] tags=[:parabolic_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "tree_1d_dgsem",
                                 "elixir_advection_diffusion_dirichlet_amr.jl"),
                        l2=[3.668679081538521e-6], linf=[0.0001053981743872842])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
end

@testitem "Parabolic1D: TreeMesh1D: elixir_advection_diffusion_neumann_amr.jl" setup=[
    Setup,
    Parabolic1D
] tags=[:parabolic_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "tree_1d_dgsem",
                                 "elixir_advection_diffusion_neumann_amr.jl"),
                        l2=[0.9974473329813947], linf=[1.0000064761980827])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
end

@testitem "Parabolic1D: TreeMesh1D: elixir_advection_diffusion.jl (AMR)" setup=[
    Setup,
    Parabolic1D
] tags=[:parabolic_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "tree_1d_dgsem",
                                 "elixir_advection_diffusion.jl"),
                        tspan=(0.0, 0.0), initial_refinement_level=5)
    tspan = (0.0, 1.0)
    ode = semidiscretize(semi, tspan)
    amr_controller = ControllerThreeLevel(semi, IndicatorMax(semi, variable = first),
                                          base_level = 4,
                                          med_level = 5, med_threshold = 0.1,
                                          max_level = 6, max_threshold = 0.6)
    amr_callback = AMRCallback(semi, amr_controller,
                               interval = 5,
                               adapt_initial_condition = true)

    # Create a CallbackSet to collect all callbacks such that they can be passed to the ODE solver
    callbacks = CallbackSet(summary_callback, analysis_callback, alive_callback,
                            amr_callback)
    sol = solve(ode, ode_alg;
                abstol = time_abs_tol, reltol = time_int_tol,
                ode_default_options()..., callback = callbacks)
    l2_error, linf_error = analysis_callback(sol)
    @test l2_error ≈ [6.487940740394583e-6]
    @test linf_error ≈ [3.262867898701227e-5]
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
end

@testitem "Parabolic1D: TreeMesh1D: elixir_advection_diffusion_implicit_sparse_jacobian.jl" setup=[
    Setup,
    Parabolic1D
] tags=[:parabolic_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "tree_1d_dgsem",
                                 "elixir_advection_diffusion_implicit_sparse_jacobian.jl"),
                        tspan=(0.0, 0.4),
                        l2=[0.05240130204342638], linf=[0.07407444680136666])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
end

@testitem "Parabolic1D: TreeMesh1D: elixir_advection_diffusion_implicit_sparse_jacobian_restart.jl" setup=[
    Setup,
    Parabolic1D
] tags=[:parabolic_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "tree_1d_dgsem",
                                 "elixir_advection_diffusion_implicit_sparse_jacobian_restart.jl"),
                        l2=[0.08292233849124372], linf=[0.11726345328639576])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
end

@testitem "Parabolic1D: elixir_advection_implicit_sparse_jacobian_restart.jl (no colorvec)" setup=[
    Setup,
    Parabolic1D
] tags=[:parabolic_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "tree_1d_dgsem",
                                 "elixir_advection_diffusion_implicit_sparse_jacobian_restart.jl"),
                        colorvec_parabolic=nothing,
                        l2=[0.08292233849124372], linf=[0.11726345328639576])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
end

@testitem "Parabolic1D: TreeMesh1D: elixir_navierstokes_convergence_periodic.jl" setup=[
    Setup,
    Parabolic1D
] tags=[:parabolic_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "tree_1d_dgsem",
                                 "elixir_navierstokes_convergence_periodic.jl"),
                        l2=[
                            0.00011333275295974374,
                            6.219651069933146e-5,
                            0.0002816727634821764
                        ],
                        linf=[
                            0.0006253615899005638,
                            0.00036185768524354955,
                            0.0016135721551933102
                        ])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
end

@testitem "Parabolic1D: TreeMesh1D: elixir_navierstokes_convergence_periodic_cfl.jl" setup=[
    Setup,
    Parabolic1D
] tags=[:parabolic_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "tree_1d_dgsem",
                                 "elixir_navierstokes_convergence_periodic_cfl.jl"),
                        l2=[
                            0.00011595398570440672,
                            6.274464247878942e-5,
                            0.00028200244192273934
                        ],
                        linf=[
                            0.0006396599382685331,
                            0.00036070632759921395,
                            0.0016361696559314964
                        ])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
end

@testitem "Parabolic1D: TreeMesh1D: elixir_navierstokes_convergence_periodic.jl: GradientVariablesEntropy" setup=[
    Setup,
    Parabolic1D
] tags=[:parabolic_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "tree_1d_dgsem",
                                 "elixir_navierstokes_convergence_periodic.jl"),
                        equations_parabolic=CompressibleNavierStokesDiffusion1D(equations,
                                                                                mu = mu(),
                                                                                Prandtl = prandtl_number(),
                                                                                gradient_variables = GradientVariablesEntropy()),
                        l2=[
                            0.00011300698739039995,
                            6.209293710944634e-5,
                            0.00028162563474383906
                        ],
                        linf=[
                            0.0006234678721388498,
                            0.00036159110715550113,
                            0.0016136182920050146
                        ])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
end

@testitem "Parabolic1D: TreeMesh1D: elixir_navierstokes_convergence_walls.jl" setup=[
    Setup,
    Parabolic1D
] tags=[:parabolic_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "tree_1d_dgsem",
                                 "elixir_navierstokes_convergence_walls.jl"),
                        l2=[
                            0.0004703758221874381,
                            0.00031848917907832425,
                            0.0014897764762842015
                        ],
                        linf=[
                            0.002999623280512509,
                            0.002844379914884066,
                            0.01268578246288321
                        ],
                        atol=1e-10)
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
end

@testitem "Parabolic1D: TreeMesh1D: elixir_navierstokes_convergence_walls.jl: GradientVariablesEntropy" setup=[
    Setup,
    Parabolic1D
] tags=[:parabolic_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "tree_1d_dgsem",
                                 "elixir_navierstokes_convergence_walls.jl"),
                        equations_parabolic=CompressibleNavierStokesDiffusion1D(equations,
                                                                                mu = mu(),
                                                                                Prandtl = prandtl_number(),
                                                                                gradient_variables = GradientVariablesEntropy()),
                        l2=[
                            0.0004599081186821817,
                            0.000320318696901367,
                            0.0015138299347089309
                        ],
                        linf=[
                            0.0027465970929201333,
                            0.0028318005097484776,
                            0.013015551256854607
                        ],
                        atol=1e-9)
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
end

@testitem "Parabolic1D: TreeMesh1D: elixir_navierstokes_convergence_walls.jl (Gauss-Legendre)" setup=[
    Setup,
    Parabolic1D
] tags=[:parabolic_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "tree_1d_dgsem",
                                 "elixir_navierstokes_convergence_walls.jl"),
                        solver=DGSEM(polydeg = 3, surface_flux = flux_hll,
                                     basis_type = GaussLegendreBasis),
                        time_int_tol=1e-10,
                        l2=[
                            4.183510913636197e-5,
                            9.605244671046932e-5,
                            0.0004207588947910475
                        ],
                        linf=[
                            0.0001529019238535323,
                            0.0004140975914396841,
                            0.0016833077670614927
                        ],
                        atol=1e-10)
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
end

@testitem "Parabolic1D: TreeMesh1D: elixir_navierstokes_convergence_walls_amr.jl" setup=[
    Setup,
    Parabolic1D
] tags=[:parabolic_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "tree_1d_dgsem",
                                 "elixir_navierstokes_convergence_walls_amr.jl"),
                        equations_parabolic=CompressibleNavierStokesDiffusion1D(equations,
                                                                                mu = mu(),
                                                                                Prandtl = prandtl_number()),
                        l2=[
                            2.4294769661376656e-5,
                            2.163375793448305e-5,
                            9.704798444764353e-5
                        ],
                        linf=[
                            0.00015581433523781385,
                            0.00018431497859248432,
                            0.0008544318777605753
                        ],
                        atol=1e-9)
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
end

@testitem "Parabolic1D: TreeMesh1D: elixir_navierstokes_convergence_walls_amr.jl: GradientVariablesEntropy" setup=[
    Setup,
    Parabolic1D
] tags=[:parabolic_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "tree_1d_dgsem",
                                 "elixir_navierstokes_convergence_walls_amr.jl"),
                        equations_parabolic=CompressibleNavierStokesDiffusion1D(equations,
                                                                                mu = mu(),
                                                                                Prandtl = prandtl_number(),
                                                                                gradient_variables = GradientVariablesEntropy()),
                        l2=[
                            2.2711261807956682e-5,
                            2.106399655146931e-5,
                            9.356923520763996e-5
                        ],
                        linf=[
                            0.00012403780156411415,
                            0.00017911182286770215,
                            0.0008654587100611622
                        ],
                        atol=1e-8)
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
end

@testitem "Parabolic1D: TreeMesh1D: elixir_navierstokes_viscous_shock.jl" setup=[
    Setup,
    Parabolic1D
] tags=[:parabolic_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "tree_1d_dgsem",
                                 "elixir_navierstokes_viscous_shock.jl"),
                        l2=[
                            0.00025762354103445303,
                            0.0001433692781569829,
                            0.00017369861968287976
                        ],
                        linf=[
                            0.0016731940030498826,
                            0.0010638575921477766,
                            0.0011495207677434394
                        ])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
end

@testitem "Parabolic1D: TreeMesh1D: elixir_navierstokes_viscous_shock.jl (Gauss-Legendre)" setup=[
    Setup,
    Parabolic1D
] tags=[:parabolic_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "tree_1d_dgsem",
                                 "elixir_navierstokes_viscous_shock.jl"),
                        solver=DGSEM(polydeg = 3, surface_flux = flux_hlle,
                                     basis_type = GaussLegendreBasis),
                        l2=[
                            0.00010415910094963455,
                            7.569570282227496e-5,
                            8.643799824895884e-5
                        ],
                        linf=[
                            0.0004795456761867989,
                            0.0003525509032139551,
                            0.0004044657250887873
                        ])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
end

@testitem "Parabolic1D: TreeMesh1D: elixir_navierstokes_viscous_shock.jl (boundary_condition_do_nothing)" setup=[
    Setup,
    Parabolic1D
] tags=[:parabolic_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "tree_1d_dgsem",
                                 "elixir_navierstokes_viscous_shock.jl"),
                        boundary_conditions_parabolic=(;
                                                       x_neg = boundary_condition_parabolic,
                                                       x_pos = boundary_condition_do_nothing),
                        l2=[
                            0.00027945595319833104,
                            0.00027552386931121406,
                            0.0005302742561529139
                        ],
                        linf=[
                            0.0016733632873879856,
                            0.001078100012167113,
                            0.0028010908633919196
                        ])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
end

@testitem "Parabolic1D: TreeMesh1D: elixir_navierstokes_viscous_shock_imex.jl" setup=[
    Setup,
    Parabolic1D
] tags=[:parabolic_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "tree_1d_dgsem",
                                 "elixir_navierstokes_viscous_shock_imex.jl"),
                        atol_lin_solve=1e-11, rtol_lin_solve=1e-10,
                        l2=[
                            0.0016637028933384878,
                            0.0014571255711373966,
                            0.0014843783212282159
                        ],
                        linf=[
                            0.00545660697650141,
                            0.003950431201790283,
                            0.004092051414554598
                        ],
                        # Relax error tols to avoid stochastic CI failures
                        atol=1e-6)
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
end

@testitem "Parabolic1D: TreeMesh1D: elixir_viscous_burgers_n_wave.jl" setup=[
    Setup,
    Parabolic1D
] tags=[:parabolic_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "tree_1d_dgsem",
                                 "elixir_viscous_burgers_n_wave.jl"),
                        l2=[0.03005971517609335], linf=[0.08174614630359545])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
end

@testitem "Parabolic1D: TreeMesh1D: elixir_viscous_burgers_shock.jl" setup=[
    Setup,
    Parabolic1D
] tags=[:parabolic_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "tree_1d_dgsem",
                                 "elixir_viscous_burgers_shock.jl"),
                        l2=[0.0025484696686361645], linf=[0.028069313915933147])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
end

@testitem "Parabolic1D: DGMulti: elixir_advection_diffusion_gradient_source_terms.jl" setup=[
    Setup,
    Parabolic1D
] tags=[:parabolic_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "dgmulti_1d",
                                 "elixir_advection_diffusion_gradient_source_terms.jl"),
                        l2=[0.01889578192611483],
                        linf=[0.03572728414418691])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
end

@testitem "Parabolic1D: DGMulti: elixir_advection_diffusion_sbp.jl" setup=[
    Setup,
    Parabolic1D
] tags=[:parabolic_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "dgmulti_1d",
                                 "elixir_advection_diffusion_sbp.jl"),
                        l2=[2.027026825559297e-5],
                        linf=[3.1997648799242384e-5])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
end

@testitem "Parabolic1D: DGMulti: elixir_navierstokes_convergence_periodic.jl" setup=[
    Setup,
    Parabolic1D
] tags=[:parabolic_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "dgmulti_1d",
                                 "elixir_navierstokes_convergence_periodic.jl"),
                        l2=[
                            3.7997410588522954e-5,
                            4.073450293422761e-5,
                            0.0002457472085334219
                        ],
                        linf=[
                            0.00010997146654223577,
                            9.179087830712973e-5,
                            0.0005468216643080837
                        ])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
end

@testitem "Parabolic1D: DGMulti: elixir_navierstokes_convergence_periodic.jl (Diff. CFL)" setup=[
    Setup,
    Parabolic1D
] tags=[:parabolic_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "dgmulti_1d",
                                 "elixir_navierstokes_convergence_periodic.jl"),
                        callbacks=CallbackSet(summary_callback, alive_callback,
                                              analysis_callback,
                                              StepsizeCallback(cfl = 0.5,
                                                               cfl_parabolic = 0.1)),
                        adaptive=false,
                        l2=[
                            3.809534312886433e-5,
                            4.072173289891623e-5,
                            0.0002457652905319161
                        ],
                        linf=[
                            0.00010927372870295216,
                            9.10595640681855e-5,
                            0.000529460047143715
                        ])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
end

@testitem "Parabolic1D: DGMulti: elixir_navierstokes_convergence_periodic.jl (GradientVariablesEntropy)" setup=[
    Setup,
    Parabolic1D
] tags=[:parabolic_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "dgmulti_1d",
                                 "elixir_navierstokes_convergence_periodic.jl"),
                        gradient_variables=GradientVariablesEntropy(),
                        l2=[
                            3.8648730222275476e-5,
                            4.0711164171890645e-5,
                            0.00024650067961444434
                        ],
                        linf=[
                            0.0001109122352365155,
                            9.172390898459781e-5,
                            0.0005473617720497259
                        ])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
end

@testitem "Parabolic1D: TreeMesh1D: elixir_diffusion_ldg.jl" setup=[Setup, Parabolic1D] tags=[:parabolic_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "tree_1d_dgsem",
                                 "elixir_diffusion_ldg.jl"),
                        initial_refinement_level=4, tspan=(0.0, 0.4), polydeg=3,
                        l2=[9.235894939144276e-6], linf=[5.402550135213957e-5])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
end

@testitem "Parabolic1D: TreeMesh1D: elixir_diffusion_ldg_newton_krylov.jl" setup=[
    Setup,
    Parabolic1D
] tags=[:parabolic_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "tree_1d_dgsem",
                                 "elixir_diffusion_ldg_newton_krylov.jl"),
                        atol_lin_solve=1e-11, rtol_lin_solve=1e-10,
                        atol_ode_solve=1e-10, rtol_ode_solve=1e-9,
                        l2=[4.14999791227157e-6], linf=[2.424658410971059e-5],
                        # Relax error tols to avoid stochastic CI failures
                        atol=1e-8)
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
end

@testitem "Parabolic1D: TreeMesh1D: elixir_diffusion_ldg_amr_boundary_layer.jl" setup=[
    Setup,
    Parabolic1D
] tags=[:parabolic_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "tree_1d_dgsem",
                                 "elixir_diffusion_ldg_amr_boundary_layer.jl"),
                        l2=[0.5881457102264551], linf=[0.9302621795999283])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)

    # Test `show` method not exercised in elixirs
    @trixi_test_nowarn show(IOContext(stdout, :compact => true), MIME"text/plain"(),
                            semi)

    # Test basic semidiscretization dispatches
    @test ndims(semi) == ndims(semi.mesh)
    @test nvariables(semi) == nvariables(semi.equations)
    @test real(semi) == real(semi.solver)

    # Test that `remake` works for `SemidiscretizationParabolic`
    semi_remade = remake(semi)
    @test semi_remade isa SemidiscretizationParabolic
    @test semi_remade !== semi
    @test semi_remade.mesh === semi.mesh
    @test Trixi.ndofsglobal(semi_remade) == Trixi.ndofsglobal(semi)
end

@testitem "Parabolic1D: TreeMesh1D consistency check: elixir_diffusion_ldg_dirichlet.jl" setup=[
    Setup,
    Parabolic1D
] tags=[:parabolic_part1] begin
    # Run the Dirichlet-Dirichlet elixir (uses `SemidiscretizationParabolic`)
    @test_trixi_include(joinpath(EXAMPLES_DIR, "tree_1d_dgsem",
                                 "elixir_diffusion_ldg_dirichlet.jl"),
                        tspan=(0.0, 0.1),
                        analysis_callback=AnalysisCallback(semi,
                                                           interval = 100,
                                                           extra_analysis_errors = (:l2_error_primitive,
                                                                                    :linf_error_primitive),
                                                           extra_analysis_integrals = (entropy,)),
                        l2=[2.3481439150004898e-6], linf=[2.4576876189230656e-5])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)

    # Store reference solution for comparison
    reference_solution = copy(sol.u[end])

    # Run again using an advection-diffusion equation with advection velocity zero
    @test_trixi_include(joinpath(EXAMPLES_DIR, "tree_1d_dgsem",
                                 "elixir_diffusion_ldg_dirichlet.jl"),
                        tspan=(0.0, 0.1),
                        equations=LinearScalarAdvectionEquation1D(0.0),
                        semi=SemidiscretizationHyperbolicParabolic(mesh,
                                                                   (equations,
                                                                    LaplaceDiffusion1D(diffusivity(),
                                                                                       equations)),
                                                                   initial_condition,
                                                                   solver;
                                                                   solver_parabolic = solver_parabolic,
                                                                   boundary_conditions = (boundary_conditions,
                                                                                          boundary_conditions)))
    # Check if the solutions for `SemidiscretizationParabolic` match those from
    # `SemidiscretizationHyperbolicParabolic` using the same Float64 tolerance defaults as
    # `@test_trixi_include` in TrixiTest.jl.
    @test sol.u[end]≈reference_solution atol=500 * eps(Float64) rtol=sqrt(eps(Float64))
end
