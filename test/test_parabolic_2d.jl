module TestExamplesParabolic2D

using Test
using Trixi

include("test_trixi.jl")

EXAMPLES_DIR = examples_dir()

# Start with a clean environment: remove Trixi.jl output directory if it exists
outdir = "out"
isdir(outdir) && rm(outdir, recursive = true)

@testset "SemidiscretizationHyperbolicParabolic (2D)" begin
#! format: noindent

@trixi_testset "DGMulti 2D rhs_parabolic!" begin
    using Trixi
    dg = DGMulti(polydeg = 2, element_type = Quad(), approximation_type = Polynomial(),
                 surface_integral = SurfaceIntegralWeakForm(flux_central),
                 volume_integral = VolumeIntegralWeakForm())
    cells_per_dimension = (2, 2)
    mesh = DGMultiMesh(dg, cells_per_dimension)

    # test with polynomial initial condition x^2 * y
    # test if we recover the exact second derivative
    initial_condition = (x, t, equations) -> SVector(x[1]^2 * x[2])

    equations = LinearScalarAdvectionEquation2D(1.0, 1.0)
    equations_parabolic = LaplaceDiffusion2D(1.0, equations)

    semi = SemidiscretizationHyperbolicParabolic(mesh, (equations, equations_parabolic),
                                                 initial_condition, dg)
    @trixi_test_nowarn show(stdout, semi)
    @trixi_test_nowarn show(stdout, MIME"text/plain"(), semi)
    @trixi_test_nowarn show(stdout, boundary_condition_do_nothing)

    @test nvariables(semi) == nvariables(equations)
    @test Base.ndims(semi) == Base.ndims(mesh)
    @test Base.real(semi) == Base.real(dg)

    ode = semidiscretize(semi, (0.0, 0.01))
    u0 = similar(ode.u0)
    Trixi.compute_coefficients!(u0, 0.0, semi)
    @test u0 ≈ ode.u0

    # test "do nothing" BC just returns first argument
    @test boundary_condition_do_nothing(u0, nothing) == u0

    (; cache, cache_parabolic, equations_parabolic) = semi
    (; gradients) = cache_parabolic
    for dim in eachindex(gradients)
        fill!(gradients[dim], zero(eltype(gradients[dim])))
    end

    # unpack VectorOfArray
    u0 = Base.parent(ode.u0)
    t = 0.0
    # pass in `boundary_condition_periodic` to skip boundary flux/integral evaluation
    parabolic_scheme = semi.solver_parabolic
    Trixi.calc_gradient!(gradients, u0, t, mesh, equations_parabolic,
                         boundary_condition_periodic, dg, parabolic_scheme,
                         cache, cache_parabolic)
    (; x, y, xq, yq) = mesh.md
    @test getindex.(gradients[1], 1) ≈ 2 * xq .* yq
    @test getindex.(gradients[2], 1) ≈ xq .^ 2

    u_flux = similar.(gradients)
    Trixi.calc_viscous_fluxes!(u_flux, u0, gradients, mesh,
                               equations_parabolic,
                               dg, cache, cache_parabolic)
    @test u_flux[1] ≈ gradients[1]
    @test u_flux[2] ≈ gradients[2]

    du = similar(u0)
    Trixi.calc_divergence!(du, u0, t, u_flux, mesh,
                           equations_parabolic,
                           boundary_condition_periodic,
                           dg, semi.solver_parabolic, cache, cache_parabolic)
    Trixi.invert_jacobian!(du, mesh, equations_parabolic, dg, cache; scaling = 1.0)
    @test getindex.(du, 1) ≈ 2 * y
end

@trixi_testset "DGMulti: elixir_advection_diffusion.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "dgmulti_2d",
                                 "elixir_advection_diffusion.jl"),
                        cells_per_dimension=(4, 4), tspan=(0.0, 0.1),
                        l2=[0.2485803335154642],
                        linf=[1.079606969242132])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)
end

@trixi_testset "DGMulti: elixir_advection_diffusion_periodic.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "dgmulti_2d",
                                 "elixir_advection_diffusion_periodic.jl"),
                        cells_per_dimension=(4, 4), tspan=(0.0, 0.1),
                        l2=[0.03180371984888462],
                        linf=[0.2136821621370909])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)
end

@trixi_testset "DGMulti: elixir_advection_diffusion_nonperiodic.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "dgmulti_2d",
                                 "elixir_advection_diffusion_nonperiodic.jl"),
                        cells_per_dimension=(4, 4), tspan=(0.0, 0.1),
                        l2=[0.002123168335604323],
                        linf=[0.00963640423513712])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)
end

@trixi_testset "DGMulti: elixir_navierstokes_convergence.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "dgmulti_2d",
                                 "elixir_navierstokes_convergence.jl"),
                        cells_per_dimension=(4, 4), tspan=(0.0, 0.1),
                        l2=[
                            0.0015355076237431118,
                            0.003384316785885901,
                            0.0036531858026850757,
                            0.009948436101649498
                        ],
                        linf=[
                            0.005522560543588462,
                            0.013425258431728926,
                            0.013962115936715924,
                            0.027483099961148838
                        ])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)
end

@trixi_testset "DGMulti: elixir_navierstokes_convergence_curved.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "dgmulti_2d",
                                 "elixir_navierstokes_convergence_curved.jl"),
                        cells_per_dimension=(4, 4), tspan=(0.0, 0.1),
                        l2=[
                            0.0042551020940351444,
                            0.011118489080358264,
                            0.011281831362358863,
                            0.035736565778376306
                        ],
                        linf=[
                            0.015071709836357083,
                            0.04103131887989486,
                            0.03990424032494211,
                            0.13094018584692968
                        ])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)
end

@trixi_testset "DGMulti: elixir_navierstokes_lid_driven_cavity.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "dgmulti_2d",
                                 "elixir_navierstokes_lid_driven_cavity.jl"),
                        cells_per_dimension=(4, 4), tspan=(0.0, 0.5),
                        l2=[
                            0.0002215612357465129,
                            0.028318325887331217,
                            0.009509168805093485,
                            0.028267893004691534
                        ],
                        linf=[
                            0.0015622793960574644,
                            0.1488665309341318,
                            0.07163235778907852,
                            0.19472797949052278
                        ])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)
end

@trixi_testset "TreeMesh2D: elixir_advection_diffusion.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "tree_2d_dgsem",
                                 "elixir_advection_diffusion.jl"),
                        initial_refinement_level=2, tspan=(0.0, 0.4), polydeg=5,
                        l2=[4.0915532997994255e-6],
                        linf=[2.3040850347877395e-5])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)
end

@trixi_testset "TreeMesh2D: elixir_advection_diffusion.jl (LDG)" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "tree_2d_dgsem",
                                 "elixir_advection_diffusion.jl"),
                        solver_parabolic=ViscousFormulationLocalDG(),
                        initial_refinement_level=2, tspan=(0.0, 0.4), polydeg=5,
                        l2=[6.193056910594806e-6], linf=[4.918855889635143e-5])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)
end

@trixi_testset "TreeMesh2D: elixir_advection_diffusion.jl (Refined mesh)" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "tree_2d_dgsem",
                                 "elixir_advection_diffusion.jl"),
                        tspan=(0.0, 0.0))
    LLID = Trixi.local_leaf_cells(mesh.tree)
    num_leaves = length(LLID)
    @assert num_leaves % 8 == 0
    Trixi.refine!(mesh.tree, LLID[1:Int(num_leaves / 8)])
    tspan = (0.0, 1.5)
    semi = SemidiscretizationHyperbolicParabolic(mesh,
                                                 (equations, equations_parabolic),
                                                 initial_condition, solver;
                                                 boundary_conditions = (boundary_conditions,
                                                                        boundary_conditions_parabolic))
    ode = semidiscretize(semi, tspan)
    analysis_callback = AnalysisCallback(semi, interval = analysis_interval)
    callbacks = CallbackSet(summary_callback, alive_callback, analysis_callback)
    sol = solve(ode, RDPK3SpFSAL49(); abstol = time_int_tol, reltol = time_int_tol,
                ode_default_options()..., callback = callbacks)
    l2_error, linf_error = analysis_callback(sol)
    @test l2_error ≈ [1.67452550744728e-6]
    @test linf_error ≈ [7.905059166368744e-6]

    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 100)
    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 100)
end

@trixi_testset "TreeMesh2D: elixir_advection_diffusion_amr.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "tree_2d_dgsem",
                                 "elixir_advection_diffusion_amr.jl"),
                        initial_refinement_level=2,
                        base_level=2,
                        med_level=3,
                        max_level=4,
                        l2=[0.0009662045510830027],
                        linf=[0.006121646998993091])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)
end

@trixi_testset "TreeMesh2D: elixir_advection_diffusion_nonperiodic.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "tree_2d_dgsem",
                                 "elixir_advection_diffusion_nonperiodic.jl"),
                        initial_refinement_level=2, tspan=(0.0, 0.1),
                        l2=[0.007646800618485118],
                        linf=[0.10067621050468958])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)
end

@trixi_testset "TreeMesh2D: elixir_advection_diffusion_nonperiodic.jl (LDG)" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "tree_2d_dgsem",
                                 "elixir_advection_diffusion_nonperiodic.jl"),
                        initial_refinement_level=2, tspan=(0.0, 0.1),
                        solver_parabolic=ViscousFormulationLocalDG(),
                        l2=[0.007009146246373517], linf=[0.09535203925012649])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)
end

@trixi_testset "TreeMesh2D: elixir_diffusion_steady_state_linear_map.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "tree_2d_dgsem",
                                 "elixir_diffusion_steady_state_linear_map.jl"),
                        l2=[2.9029827892716424e-5], linf=[0.0003022506331279151])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)
end

@trixi_testset "TreeMesh2D: elixir_navierstokes_convergence.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "tree_2d_dgsem",
                                 "elixir_navierstokes_convergence.jl"),
                        initial_refinement_level=2, tspan=(0.0, 0.1),
                        analysis_callback=AnalysisCallback(semi,
                                                           interval = analysis_interval,
                                                           extra_analysis_integrals = (energy_kinetic,
                                                                                       energy_internal,
                                                                                       enstrophy)),
                        l2=[
                            0.0021116725306624543,
                            0.003432235149083229,
                            0.003874252819605527,
                            0.012469246082535005
                        ],
                        linf=[
                            0.012006418939279007,
                            0.03552087120962882,
                            0.02451274749189282,
                            0.11191122588626357
                        ])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)
end

@trixi_testset "TreeMesh2D: elixir_navierstokes_convergence.jl (isothermal walls)" begin
    using Trixi: Trixi
    @test_trixi_include(joinpath(EXAMPLES_DIR, "tree_2d_dgsem",
                                 "elixir_navierstokes_convergence.jl"),
                        initial_refinement_level=2, tspan=(0.0, 0.1),
                        heat_bc_top_bottom=Isothermal((x, t, equations) -> Trixi.temperature(initial_condition_navier_stokes_convergence_test(x,
                                                                                                                                              t,
                                                                                                                                              equations),
                                                                                             equations)),
                        l2=[
                            0.0021036296503840883,
                            0.003435843933397192,
                            0.003867359878114748,
                            0.012670355349293195
                        ],
                        linf=[
                            0.01200626179308184,
                            0.03550212518997239,
                            0.025107947320178275,
                            0.11647078036751068
                        ])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)
end

@trixi_testset "TreeMesh2D: elixir_navierstokes_convergence.jl (Entropy gradient variables)" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "tree_2d_dgsem",
                                 "elixir_navierstokes_convergence.jl"),
                        initial_refinement_level=2, tspan=(0.0, 0.1),
                        gradient_variables=GradientVariablesEntropy(),
                        l2=[
                            0.002140374251729127,
                            0.003425828709496601,
                            0.0038915122887358097,
                            0.012506862342858291
                        ],
                        linf=[
                            0.012244412004772665,
                            0.03507559186131113,
                            0.02458089234472249,
                            0.11425600758024679
                        ])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)
end

@trixi_testset "TreeMesh2D: elixir_navierstokes_convergence.jl (Entropy gradient variables, isothermal walls)" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "tree_2d_dgsem",
                                 "elixir_navierstokes_convergence.jl"),
                        initial_refinement_level=2, tspan=(0.0, 0.1),
                        gradient_variables=GradientVariablesEntropy(),
                        heat_bc_top_bottom=Isothermal((x, t, equations) -> Trixi.temperature(initial_condition_navier_stokes_convergence_test(x,
                                                                                                                                              t,
                                                                                                                                              equations),
                                                                                             equations)),
                        l2=[
                            0.0021349737347923716,
                            0.0034301388278178365,
                            0.0038928324473968836,
                            0.012693611436338
                        ],
                        linf=[
                            0.012244236275761766,
                            0.03505406631430898,
                            0.025099598505644406,
                            0.11795616324985403
                        ])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)
end

@trixi_testset "TreeMesh2D: elixir_navierstokes_convergence.jl (flux differencing)" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "tree_2d_dgsem",
                                 "elixir_navierstokes_convergence.jl"),
                        initial_refinement_level=2, tspan=(0.0, 0.1),
                        volume_integral=VolumeIntegralFluxDifferencing(flux_central),
                        l2=[
                            0.0021116725306612075,
                            0.0034322351490838703,
                            0.0038742528196011594,
                            0.012469246082545557
                        ],
                        linf=[
                            0.012006418939262131,
                            0.0355208712096602,
                            0.024512747491999436,
                            0.11191122588669522
                        ])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)
end

@trixi_testset "TreeMesh2D: elixir_navierstokes_convergence.jl (Refined mesh)" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "tree_2d_dgsem",
                                 "elixir_navierstokes_convergence.jl"),
                        tspan=(0.0, 0.0), initial_refinement_level=3)
    LLID = Trixi.local_leaf_cells(mesh.tree)
    num_leaves = length(LLID)
    @assert num_leaves % 4 == 0
    Trixi.refine!(mesh.tree, LLID[1:Int(num_leaves / 4)])
    tspan = (0.0, 0.5)
    semi = SemidiscretizationHyperbolicParabolic(mesh, (equations, equations_parabolic),
                                                 initial_condition, solver;
                                                 boundary_conditions = (boundary_conditions,
                                                                        boundary_conditions_parabolic),
                                                 source_terms = source_terms_navier_stokes_convergence_test)
    ode = semidiscretize(semi, tspan)
    analysis_callback = AnalysisCallback(semi, interval = analysis_interval)
    callbacks = CallbackSet(summary_callback, alive_callback, analysis_callback)
    sol = solve(ode, RDPK3SpFSAL49(); abstol = time_int_tol, reltol = time_int_tol,
                dt = 1e-5,
                ode_default_options()..., callback = callbacks)
    l2_error, linf_error = analysis_callback(sol)
    @test l2_error ≈
          [0.00024296959174050973;
           0.00020932631586399853;
           0.0005390572390981241;
           0.00026753561391316933]
    @test linf_error ≈
          [0.0016210102053486608;
           0.0025932876486537016;
           0.0029539073438284817;
           0.0020771191202548778]
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)
end

@trixi_testset "TreeMesh2D: elixir_navierstokes_lid_driven_cavity.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "tree_2d_dgsem",
                                 "elixir_navierstokes_lid_driven_cavity.jl"),
                        initial_refinement_level=2, tspan=(0.0, 0.5),
                        l2=[
                            0.00015144571529699053,
                            0.018766076072331623,
                            0.007065070765652574,
                            0.0208399005734258
                        ],
                        linf=[
                            0.0014523369373669048,
                            0.12366779944955864,
                            0.05532450997115432,
                            0.16099927805328207
                        ])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)
end

@trixi_testset "TreeMesh2D: elixir_navierstokes_shearlayer_amr.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "tree_2d_dgsem",
                                 "elixir_navierstokes_shearlayer_amr.jl"),
                        l2=[
                            0.005155557460409018,
                            0.4048446934219344,
                            0.43040068852937047,
                            1.1255130552079322
                        ],
                        linf=[
                            0.03287305649809613,
                            1.1656793717431393,
                            1.3917196016246969,
                            8.146587380114653
                        ],
                        tspan=(0.0, 0.7))
end

@trixi_testset "TreeMesh2D: elixir_navierstokes_taylor_green_vortex_sutherland.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "tree_2d_dgsem",
                                 "elixir_navierstokes_taylor_green_vortex_sutherland.jl"),
                        l2=[
                            0.001452856280034929,
                            0.0007538775539989481,
                            0.0007538775539988681,
                            0.011035506549989587
                        ],
                        linf=[
                            0.003291912841311362,
                            0.002986462478096974,
                            0.0029864624780958637,
                            0.0231954665514138
                        ],
                        tspan=(0.0, 1.0))
end

@trixi_testset "P4estMesh2D: elixir_advection_diffusion_periodic.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "p4est_2d_dgsem",
                                 "elixir_advection_diffusion_periodic.jl"),
                        trees_per_dimension=(1, 1), initial_refinement_level=2,
                        tspan=(0.0, 0.5),
                        l2=[0.0023754695605828443],
                        linf=[0.008154128363741964])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)
end

@trixi_testset "P4estMesh2D: elixir_advection_diffusion_periodic.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "p4est_2d_dgsem",
                                 "elixir_advection_diffusion_periodic.jl"),
                        trees_per_dimension=(1, 1), initial_refinement_level=2,
                        tspan=(0.0, 0.5),
                        l2=[0.0023754695605828443],
                        linf=[0.008154128363741964])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)
end

@trixi_testset "P4estMesh2D: elixir_advection_diffusion_periodic_curved.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "p4est_2d_dgsem",
                                 "elixir_advection_diffusion_periodic_curved.jl"),
                        trees_per_dimension=(1, 1), initial_refinement_level=2,
                        tspan=(0.0, 0.5),
                        l2=[0.006708147442490916],
                        linf=[0.04807038397976693])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)
end

@trixi_testset "P4estMesh2D: elixir_advection_diffusion_periodic_amr.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "p4est_2d_dgsem",
                                 "elixir_advection_diffusion_periodic_amr.jl"),
                        tspan=(0.0, 0.01),
                        l2=[0.014715887539773128],
                        linf=[0.2285802791900049])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)
end

@trixi_testset "P4estMesh2D: elixir_advection_diffusion_nonperiodic_amr.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "p4est_2d_dgsem",
                                 "elixir_advection_diffusion_nonperiodic_amr.jl"),
                        tspan=(0.0, 0.01),
                        l2=[0.007934195641974433],
                        linf=[0.11030265194954081])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)
end

@trixi_testset "P4estMesh2D: elixir_advection_diffusion_nonperiodic_curved.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "p4est_2d_dgsem",
                                 "elixir_advection_diffusion_nonperiodic_curved.jl"),
                        trees_per_dimension=(1, 1), initial_refinement_level=2,
                        tspan=(0.0, 0.5),
                        l2=[0.00919917034843865],
                        linf=[0.14186297438393505])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)
end

@trixi_testset "P4estMesh2D: elixir_navierstokes_convergence.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "p4est_2d_dgsem",
                                 "elixir_navierstokes_convergence.jl"),
                        initial_refinement_level=1, tspan=(0.0, 0.2),
                        l2=[
                            0.0003811978986531135,
                            0.0005874314969137914,
                            0.0009142898787681551,
                            0.0011613918893790497
                        ],
                        linf=[
                            0.0021633623985426453,
                            0.009484348273965089,
                            0.0042315720663082534,
                            0.011661660264076446
                        ])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)
end

@trixi_testset "P4estMesh2D: elixir_navierstokes_convergence_nonperiodic.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "p4est_2d_dgsem",
                                 "elixir_navierstokes_convergence_nonperiodic.jl"),
                        initial_refinement_level=1, tspan=(0.0, 0.2),
                        l2=[
                            0.0004036496258545996,
                            0.0005869762480189079,
                            0.0009148853742181908,
                            0.0011984191532764543
                        ],
                        linf=[
                            0.0024993634989209923,
                            0.009487866203496731,
                            0.004505829506103787,
                            0.011634902753554499
                        ])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)
end

@trixi_testset "P4estMesh2D: elixir_navierstokes_lid_driven_cavity.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "p4est_2d_dgsem",
                                 "elixir_navierstokes_lid_driven_cavity.jl"),
                        initial_refinement_level=2, tspan=(0.0, 0.5),
                        l2=[
                            0.00028716166408816073,
                            0.08101204560401647,
                            0.02099595625377768,
                            0.05008149754143295
                        ],
                        linf=[
                            0.014804500261322406,
                            0.9513271652357098,
                            0.7223919625994717,
                            1.4846907331004786
                        ])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)
end

@trixi_testset "P4estMesh2D: elixir_navierstokes_lid_driven_cavity_amr.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "p4est_2d_dgsem",
                                 "elixir_navierstokes_lid_driven_cavity_amr.jl"),
                        tspan=(0.0, 1.0),
                        l2=[
                            0.0005323841980601085, 0.07892044543547208,
                            0.02909671646389337, 0.11717468256112017
                        ],
                        linf=[
                            0.006045292737899444, 0.9233292581786228,
                            0.7982129977236198, 1.6864546235292153
                        ])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)
end

@trixi_testset "elixir_navierstokes_NACA0012airfoil_mach08.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "p4est_2d_dgsem",
                                 "elixir_navierstokes_NACA0012airfoil_mach08.jl"),
                        l2=[0.000186486564226516,
                            0.0005076712323400374,
                            0.00038074588984354107,
                            0.002128177239782089],
                        linf=[0.5153387072802718,
                            1.199362305026636,
                            0.9077214424040279,
                            5.666071182328691], tspan=(0.0, 0.001),
                        initial_refinement_level=0,)
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)

    u_ode = copy(sol.u[end])
    du_ode = zero(u_ode) # Just a placeholder in this case

    u = Trixi.wrap_array(u_ode, semi)
    du = Trixi.wrap_array(du_ode, semi)

    drag_p = Trixi.analyze(drag_coefficient, du, u, tspan[2], mesh, equations, solver,
                           semi.cache, semi)
    lift_p = Trixi.analyze(lift_coefficient, du, u, tspan[2], mesh, equations, solver,
                           semi.cache, semi)

    drag_f = Trixi.analyze(drag_coefficient_shear_force, du, u, tspan[2], mesh,
                           equations, equations_parabolic, solver,
                           semi.cache, semi, semi.cache_parabolic)
    lift_f = Trixi.analyze(lift_coefficient_shear_force, du, u, tspan[2], mesh,
                           equations, equations_parabolic, solver,
                           semi.cache, semi, semi.cache_parabolic)

    @test isapprox(drag_p, 0.17963843913309516, atol = 1e-13)
    @test isapprox(lift_p, 0.26462588007949367, atol = 1e-13)

    @test isapprox(drag_f, 1.5427441885921553, atol = 1e-13)
    @test isapprox(lift_f, 0.005621910087395724, atol = 1e-13)
end

@trixi_testset "elixir_navierstokes_NACA0012airfoil_mach085_restart.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "p4est_2d_dgsem",
                                 "elixir_navierstokes_NACA0012airfoil_mach085_restart.jl"),
                        l2=[
                            6.191672324705442e-6,
                            0.00011583392224949682,
                            0.00011897020463459889,
                            0.006467379086802275
                        ],
                        linf=[
                            0.0017446176443216936,
                            0.06961708834164942,
                            0.037063246278530367,
                            1.4435072005258793
                        ], tspan=(0.0, 0.01),)
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)
end

@trixi_testset "P4estMesh2D: elixir_navierstokes_viscous_shock.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "p4est_2d_dgsem",
                                 "elixir_navierstokes_viscous_shock.jl"),
                        l2=[
                            0.0002576236264053728,
                            0.00014336949098706463,
                            7.189100338239794e-17,
                            0.00017369905124642074
                        ],
                        linf=[
                            0.0016731940983241156,
                            0.0010638640749656147,
                            5.59044079947959e-16,
                            0.001149532023891009
                        ])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)
end

@trixi_testset "P4estMesh2D: elixir_navierstokes_viscous_shock_newton_krylov.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "p4est_2d_dgsem",
                                 "elixir_navierstokes_viscous_shock_newton_krylov.jl"),
                        tspan=(0.0, 0.1),
                        atol_lin_solve=1e-11,
                        rtol_lin_solve=1e-11,
                        atol_ode_solve=1e-10,
                        rtol_ode_solve=1e-10,
                        l2=[
                            3.428501006908931e-5,
                            2.5967418005884837e-5,
                            2.7084890458524478e-17,
                            2.855861765163304e-5
                        ],
                        linf=[
                            0.00018762342908784646,
                            0.0001405900207752664,
                            3.661971738081151e-16,
                            0.00014510700486747297
                        ])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)
end

@trixi_testset "elixir_navierstokes_SD7003airfoil.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "p4est_2d_dgsem",
                                 "elixir_navierstokes_SD7003airfoil.jl"),
                        l2=[
                            9.292899618740586e-5,
                            0.0001350510200255721,
                            7.964907891113045e-5,
                            0.0002336568736996096
                        ],
                        linf=[
                            0.2845637352223691,
                            0.295808392241858,
                            0.19309201225626166,
                            0.7188927326929244
                        ],
                        tspan=(0.0, 5e-3))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)
end

@trixi_testset "elixir_navierstokes_SD7003airfoil.jl (CFL-Interval)" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "p4est_2d_dgsem",
                                 "elixir_navierstokes_SD7003airfoil.jl"),
                        l2=[
                            9.292895651912815e-5,
                            0.0001350510066877861,
                            7.964905098170568e-5,
                            0.00023365678706785303
                        ],
                        linf=[
                            0.2845614660523972,
                            0.29577255454711177,
                            0.19307666048254143,
                            0.7188872358580256
                        ],
                        tspan=(0.0, 5e-3),
                        stepsize_callback=StepsizeCallback(cfl = 2.2, interval = 5))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)
end

@trixi_testset "elixir_navierstokes_vortex_street.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "p4est_2d_dgsem",
                                 "elixir_navierstokes_vortex_street.jl"),
                        l2=[
                            0.012420217727434794,
                            0.028935260981567217,
                            0.023078384429351353,
                            0.11317643179072025
                        ],
                        linf=[
                            0.4484833725983406,
                            1.268913882714608,
                            0.7071821629898418,
                            3.643975012834931
                        ],
                        tspan=(0.0, 1.0))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)
end

@trixi_testset "elixir_navierstokes_poiseuille_flow.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "p4est_2d_dgsem",
                                 "elixir_navierstokes_poiseuille_flow.jl"),
                        l2=[
                            0.028671228188785286,
                            0.2136420195921885,
                            0.009953689550858224,
                            0.13216036594768157
                        ],
                        linf=[
                            0.30901218409540543,
                            1.3488655161645846,
                            0.1304661713119874,
                            1.2094591729756736],
                        tspan=(0.0, 1.0))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)
end

@trixi_testset "elixir_navierstokes_kelvin_helmholtz_instability_sc_subcell.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "tree_2d_dgsem",
                                 "elixir_navierstokes_kelvin_helmholtz_instability_sc_subcell.jl"),
                        l2=[
                            0.1987691550257618,
                            0.1003336666735962,
                            0.1599420846677608,
                            0.07314642823482713
                        ],
                        linf=[
                            0.8901520920065688,
                            0.47421178500575756,
                            0.38859478648621326,
                            0.3247497921546598
                        ],
                        tspan=(0.0, 1.0))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    # Larger values for allowed allocations due to usage of custom
    # integrator which are not *recorded* for the methods from
    # OrdinaryDiffEq.jl
    # Corresponding issue: https://github.com/trixi-framework/Trixi.jl/issues/1877
    @test_allocations(Trixi.rhs!, semi, sol, 15000)
end

@trixi_testset "elixir_navierstokes_freestream_symmetry.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "p4est_2d_dgsem",
                                 "elixir_navierstokes_freestream_symmetry.jl"),
                        l2=[
                            4.37868326434923e-15,
                            7.002449644031901e-16,
                            1.0986677074164136e-14,
                            1.213800745067394e-14
                        ],
                        linf=[
                            2.531308496145357e-14,
                            3.8367543336926215e-15,
                            4.9960036108132044e-14,
                            6.705747068735946e-14
                        ])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)
end

@trixi_testset "elixir_navierstokes_couette_flow.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "p4est_2d_dgsem",
                                 "elixir_navierstokes_couette_flow.jl"),
                        l2=[
                            0.009585252225488753,
                            0.007939233099864973,
                            0.0007617512688442657,
                            0.027229870237669436
                        ],
                        linf=[
                            0.027230029149270862,
                            0.027230451118692933,
                            0.0038642959675975713,
                            0.04738248734987671
                        ])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)
end

@trixi_testset "elixir_navierstokes_blast_reflective.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "p4est_2d_dgsem",
                                 "elixir_navierstokes_blast_reflective.jl"),
                        l2=[
                            0.013077652405653456,
                            0.03267271241679693,
                            0.03267271241679689,
                            0.19993587690609887
                        ],
                        linf=[
                            0.232863088636711,
                            0.5958991303183211,
                            0.5958991303183204,
                            3.0621202120365467
                        ],
                        tspan=(0.0, 0.01),
                        sol=solve(ode, ode_alg;
                                  adaptive = false, dt = 1e-4,
                                  ode_default_options()..., callback = callbacks))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)
end
end

# Clean up afterwards: delete Trixi.jl output directory
@test_nowarn isdir(outdir) && rm(outdir, recursive = true)

end # module
