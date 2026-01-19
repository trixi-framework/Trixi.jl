module TestExamplesParabolic1D

using Test
using Trixi

include("test_trixi.jl")

EXAMPLES_DIR = examples_dir()

# Start with a clean environment: remove Trixi output directory if it exists
outdir = "out"
isdir(outdir) && rm(outdir, recursive = true)

@testset "SemidiscretizationHyperbolicParabolic (1D)" begin
#! format: noindent

@trixi_testset "TreeMesh1D: elixir_advection_diffusion.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "tree_1d_dgsem",
                                 "elixir_advection_diffusion.jl"),
                        initial_refinement_level=4, tspan=(0.0, 0.4), polydeg=3,
                        l2=[8.40483031802723e-6],
                        linf=[2.8990878868540015e-5])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)
    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
end

@trixi_testset "TreeMesh1D: elixir_advection_diffusion_ldg.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "tree_1d_dgsem",
                                 "elixir_advection_diffusion_ldg.jl"),
                        initial_refinement_level=4, tspan=(0.0, 0.4), polydeg=3,
                        l2=[9.234438322146518e-6], linf=[5.425491770139068e-5])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)
    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
end

@trixi_testset "TreeMesh1D: elixir_advection_diffusion_gradient_source_terms.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "tree_1d_dgsem",
                                 "elixir_advection_diffusion_gradient_source_terms.jl"),
                        initial_refinement_level=4, tspan=(0.0, 0.4), polydeg=3,
                        l2=[1.0990454698899562e-5], linf=[6.469747978055107e-5])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)
    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
end

@trixi_testset "TreeMesh1D: elixir_diffusion_ldg.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "tree_1d_dgsem",
                                 "elixir_diffusion_ldg.jl"),
                        initial_refinement_level=4, tspan=(0.0, 0.4), polydeg=3,
                        l2=[9.235894939144276e-6], linf=[5.402550135213957e-5])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)
end

@trixi_testset "TreeMesh1D: elixir_diffusion_ldg_newton_krylov.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "tree_1d_dgsem",
                                 "elixir_diffusion_ldg_newton_krylov.jl"),
                        atol_lin_solve=1e-11, rtol_lin_solve=1e-10,
                        atol_ode_solve=1e-10, rtol_ode_solve=1e-9,
                        l2=[4.14999791227157e-6], linf=[2.424658410971059e-5])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)
    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
end

@trixi_testset "TreeMesh1D: elixir_advection_diffusion_restart.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "tree_1d_dgsem",
                                 "elixir_advection_diffusion_restart.jl"),
                        l2=[1.0679933947301556e-5],
                        linf=[3.910500545667439e-5])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)
end

@trixi_testset "TreeMesh1D: elixir_advection_diffusion_cfl.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "tree_1d_dgsem",
                                 "elixir_advection_diffusion_cfl.jl"),
                        l2=[6.763177530985864e-5], linf=[0.0002344578097126515])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)
    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
end

@trixi_testset "TreeMesh1D: elixir_advection_diffusion_dirichlet_amr.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "tree_1d_dgsem",
                                 "elixir_advection_diffusion_dirichlet_amr.jl"),
                        l2=[3.668679081538521e-6], linf=[0.0001053981743872842])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)
    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
end

@trixi_testset "TreeMesh1D: elixir_advection_diffusion_neumann_amr.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "tree_1d_dgsem",
                                 "elixir_advection_diffusion_neumann_amr.jl"),
                        l2=[0.9974473329813947], linf=[1.0000064761980827])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)
    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
end

@trixi_testset "TreeMesh1D: elixir_advection_diffusion.jl (AMR)" begin
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
    @test_allocations(Trixi.rhs!, semi, sol, 1000)
    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
end

@trixi_testset "TreeMesh1D: elixir_advection_diffusion_implicit_sparse_jacobian.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "tree_1d_dgsem",
                                 "elixir_advection_diffusion_implicit_sparse_jacobian.jl"),
                        tspan=(0.0, 0.4),
                        l2=[0.05240130204342638], linf=[0.07407444680136666])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)
    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
end

@trixi_testset "TreeMesh1D: elixir_advection_diffusion_implicit_sparse_jacobian_restart.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "tree_1d_dgsem",
                                 "elixir_advection_diffusion_implicit_sparse_jacobian_restart.jl"),
                        l2=[0.08292233849124372], linf=[0.11726345328639576])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)
    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
end

@trixi_testset "elixir_advection_implicit_sparse_jacobian_restart.jl (no colorvec)" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "tree_1d_dgsem",
                                 "elixir_advection_diffusion_implicit_sparse_jacobian_restart.jl"),
                        colorvec_parabolic=nothing,
                        l2=[0.08292233849124372], linf=[0.11726345328639576])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)
    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
end

@trixi_testset "TreeMesh1D: elixir_navierstokes_convergence_periodic.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "tree_1d_dgsem",
                                 "elixir_navierstokes_convergence_periodic.jl"),
                        l2=[
                            0.0001133835907077494,
                            6.226282245610444e-5,
                            0.0002820171699999139
                        ],
                        linf=[
                            0.0006255102377159538,
                            0.00036195501456059986,
                            0.0016147729485886941
                        ])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)
    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
end

@trixi_testset "TreeMesh1D: elixir_navierstokes_convergence_periodic_cfl.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "tree_1d_dgsem",
                                 "elixir_navierstokes_convergence_periodic_cfl.jl"),
                        l2=[
                            0.00011582226718630047,
                            6.277345250542003e-5,
                            0.0002822257163816253
                        ],
                        linf=[
                            0.0006389893469918029,
                            0.0003608325914101762,
                            0.0016369657641206459
                        ])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)
    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
end

@trixi_testset "TreeMesh1D: elixir_navierstokes_convergence_periodic.jl: GradientVariablesEntropy" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "tree_1d_dgsem",
                                 "elixir_navierstokes_convergence_periodic.jl"),
                        equations_parabolic=CompressibleNavierStokesDiffusion1D(equations,
                                                                                mu = mu(),
                                                                                Prandtl = prandtl_number(),
                                                                                gradient_variables = GradientVariablesEntropy()),
                        l2=[
                            0.00011310615871043463,
                            6.216495207074201e-5,
                            0.00028195843110817814
                        ],
                        linf=[
                            0.0006240837363233886,
                            0.0003616694320713876,
                            0.0016147339542413874
                        ])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)
    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
end

@trixi_testset "TreeMesh1D: elixir_navierstokes_convergence_walls.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "tree_1d_dgsem",
                                 "elixir_navierstokes_convergence_walls.jl"),
                        l2=[
                            0.0004702331100298379,
                            0.0003218173539588441,
                            0.001496626616191212
                        ],
                        linf=[
                            0.0029963751636357117,
                            0.0028639041695096433,
                            0.012691132694550689
                        ],
                        atol=1e-10)
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)
    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
end

@trixi_testset "TreeMesh1D: elixir_navierstokes_convergence_walls.jl: GradientVariablesEntropy" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "tree_1d_dgsem",
                                 "elixir_navierstokes_convergence_walls.jl"),
                        equations_parabolic=CompressibleNavierStokesDiffusion1D(equations,
                                                                                mu = mu(),
                                                                                Prandtl = prandtl_number(),
                                                                                gradient_variables = GradientVariablesEntropy()),
                        l2=[
                            0.00046085004909354776,
                            0.0003243109084492897,
                            0.0015159733164383632
                        ],
                        linf=[
                            0.0027548031865172184,
                            0.0028567713569609024,
                            0.012941793735691931
                        ],
                        atol=1e-9)
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)
    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
end

@trixi_testset "TreeMesh1D: elixir_navierstokes_convergence_walls_amr.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "tree_1d_dgsem",
                                 "elixir_navierstokes_convergence_walls_amr.jl"),
                        equations_parabolic=CompressibleNavierStokesDiffusion1D(equations,
                                                                                mu = mu(),
                                                                                Prandtl = prandtl_number()),
                        l2=[
                            2.5278845598681636e-5,
                            2.5540145802666872e-5,
                            0.0001211867535580826
                        ],
                        linf=[
                            0.0001466387202588848,
                            0.00019422419092429135,
                            0.0009556449835592673
                        ],
                        atol=1e-9)
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)
    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
end

@trixi_testset "TreeMesh1D: elixir_navierstokes_convergence_walls_amr.jl: GradientVariablesEntropy" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "tree_1d_dgsem",
                                 "elixir_navierstokes_convergence_walls_amr.jl"),
                        equations_parabolic=CompressibleNavierStokesDiffusion1D(equations,
                                                                                mu = mu(),
                                                                                Prandtl = prandtl_number(),
                                                                                gradient_variables = GradientVariablesEntropy()),
                        l2=[
                            2.4593521887223632e-5,
                            2.3928212900127102e-5,
                            0.00011252332663824173
                        ],
                        linf=[
                            0.00011850494672183132,
                            0.00018987676556476442,
                            0.0009597423024825247
                        ],
                        atol=1e-9)
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)
    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
end

@trixi_testset "TreeMesh1D: elixir_navierstokes_viscous_shock.jl" begin
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
    @test_allocations(Trixi.rhs!, semi, sol, 1000)
    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
end

@trixi_testset "TreeMesh1D: elixir_navierstokes_viscous_shock_imex.jl" begin
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
                        atol=1e-7)
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)
    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
end

@trixi_testset "DGMulti: elixir_advection_diffusion_gradient_source_terms.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "dgmulti_1d",
                                 "elixir_advection_diffusion_gradient_source_terms.jl"),
                        l2=[0.01889578192611483],
                        linf=[0.03572728414418691])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)
    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
end

@trixi_testset "DGMulti: elixir_advection_diffusion_sbp.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "dgmulti_1d",
                                 "elixir_advection_diffusion_sbp.jl"),
                        l2=[2.027026825559297e-5],
                        linf=[3.1997648799242384e-5])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)
    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
end

@trixi_testset "DGMulti: elixir_navierstokes_convergence_periodic.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "dgmulti_1d",
                                 "elixir_navierstokes_convergence_periodic.jl"),
                        l2=[
                            3.7943372542675425e-5,
                            4.078766566292102e-5,
                            0.00024524952267207235
                        ],
                        linf=[
                            0.00010969455084941515,
                            9.183113730193426e-5,
                            0.0005450421812014383
                        ])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)
end

@trixi_testset "DGMulti: elixir_navierstokes_convergence_periodic.jl (Diff. CFL)" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "dgmulti_1d",
                                 "elixir_navierstokes_convergence_periodic.jl"),
                        callbacks=CallbackSet(summary_callback, alive_callback,
                                              analysis_callback,
                                              StepsizeCallback(cfl = 0.5,
                                                               cfl_diffusive = 0.1)),
                        adaptive=false,
                        l2=[
                            3.804624387087144e-5,
                            4.0776239664045585e-5,
                            0.0002452796554181002
                        ],
                        linf=[
                            0.00010899905841177393,
                            9.108558032178138e-5,
                            0.0005277952647766426
                        ])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)
    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
end

@trixi_testset "DGMulti: elixir_navierstokes_convergence_periodic.jl (GradientVariablesEntropy)" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "dgmulti_1d",
                                 "elixir_navierstokes_convergence_periodic.jl"),
                        gradient_variables=GradientVariablesEntropy(),
                        l2=[
                            3.855011159752911e-5,
                            4.077230736638483e-5,
                            0.0002457818746735199
                        ],
                        linf=[
                            0.00011052974882530542,
                            9.179337892284423e-5,
                            0.00054534178933352
                        ])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)
    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
end
end

# Clean up afterwards: delete Trixi output directory
@test_nowarn isdir(outdir) && rm(outdir, recursive = true)

end # module
