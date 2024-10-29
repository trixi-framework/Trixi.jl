module TestExamplesParabolic2D

using Test
using Trixi

include("test_trixi.jl")

# Start with a clean environment: remove Trixi.jl output directory if it exists
outdir = "out"
isdir(outdir) && rm(outdir, recursive = true)

@testset "SemidiscretizationHyperbolicParabolic (2D)" begin
#! format: noindent

@trixi_testset "DGMulti 2D rhs_parabolic!" begin
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

    semi = SemidiscretizationHyperbolicParabolic(mesh, equations, equations_parabolic,
                                                 initial_condition, dg)
    @test_nowarn_mod show(stdout, semi)
    @test_nowarn_mod show(stdout, MIME"text/plain"(), semi)
    @test_nowarn_mod show(stdout, boundary_condition_do_nothing)

    @test nvariables(semi) == nvariables(equations)
    @test Base.ndims(semi) == Base.ndims(mesh)
    @test Base.real(semi) == Base.real(dg)

    ode = semidiscretize(semi, (0.0, 0.01))
    u0 = similar(ode.u0)
    Trixi.compute_coefficients!(u0, 0.0, semi)
    @test u0 ≈ ode.u0

    # test "do nothing" BC just returns first argument
    @test boundary_condition_do_nothing(u0, nothing) == u0

    @unpack cache, cache_parabolic, equations_parabolic = semi
    @unpack gradients = cache_parabolic
    for dim in eachindex(gradients)
        fill!(gradients[dim], zero(eltype(gradients[dim])))
    end

    t = 0.0
    # pass in `boundary_condition_periodic` to skip boundary flux/integral evaluation
    Trixi.calc_gradient!(gradients, ode.u0, t, mesh, equations_parabolic,
                         boundary_condition_periodic, dg, cache, cache_parabolic)
    @unpack x, y, xq, yq = mesh.md
    @test getindex.(gradients[1], 1) ≈ 2 * xq .* yq
    @test getindex.(gradients[2], 1) ≈ xq .^ 2

    u_flux = similar.(gradients)
    Trixi.calc_viscous_fluxes!(u_flux, ode.u0, gradients, mesh, equations_parabolic,
                               dg, cache, cache_parabolic)
    @test u_flux[1] ≈ gradients[1]
    @test u_flux[2] ≈ gradients[2]

    du = similar(ode.u0)
    Trixi.calc_divergence!(du, ode.u0, t, u_flux, mesh, equations_parabolic,
                           boundary_condition_periodic,
                           dg, semi.solver_parabolic, cache, cache_parabolic)
    @test getindex.(du, 1) ≈ 2 * y
end

@trixi_testset "DGMulti: elixir_advection_diffusion.jl" begin
    @test_trixi_include(joinpath(examples_dir(), "dgmulti_2d",
                                 "elixir_advection_diffusion.jl"),
                        cells_per_dimension=(4, 4), tspan=(0.0, 0.1),
                        l2=[0.2485803335154642],
                        linf=[1.079606969242132])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "DGMulti: elixir_advection_diffusion_periodic.jl" begin
    @test_trixi_include(joinpath(examples_dir(), "dgmulti_2d",
                                 "elixir_advection_diffusion_periodic.jl"),
                        cells_per_dimension=(4, 4), tspan=(0.0, 0.1),
                        l2=[0.03180371984888462],
                        linf=[0.2136821621370909])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "DGMulti: elixir_advection_diffusion_nonperiodic.jl" begin
    @test_trixi_include(joinpath(examples_dir(), "dgmulti_2d",
                                 "elixir_advection_diffusion_nonperiodic.jl"),
                        cells_per_dimension=(4, 4), tspan=(0.0, 0.1),
                        l2=[0.002123168335604323],
                        linf=[0.00963640423513712])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "DGMulti: elixir_navierstokes_convergence.jl" begin
    @test_trixi_include(joinpath(examples_dir(), "dgmulti_2d",
                                 "elixir_navierstokes_convergence.jl"),
                        cells_per_dimension=(4, 4), tspan=(0.0, 0.1),
                        l2=[
                            0.0015355076812510957,
                            0.0033843168272696756,
                            0.0036531858107443434,
                            0.009948436427519214
                        ],
                        linf=[
                            0.005522560467190019,
                            0.013425258500730508,
                            0.013962115643482154,
                            0.027483102120502423
                        ])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "DGMulti: elixir_navierstokes_convergence_curved.jl" begin
    @test_trixi_include(joinpath(examples_dir(), "dgmulti_2d",
                                 "elixir_navierstokes_convergence_curved.jl"),
                        cells_per_dimension=(4, 4), tspan=(0.0, 0.1),
                        l2=[
                            0.004255101916146187,
                            0.011118488923215765,
                            0.011281831283462686,
                            0.03573656447388509
                        ],
                        linf=[
                            0.015071710669706473,
                            0.04103132025858458,
                            0.03990424085750277,
                            0.1309401718598764
                        ],)
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "DGMulti: elixir_navierstokes_lid_driven_cavity.jl" begin
    @test_trixi_include(joinpath(examples_dir(), "dgmulti_2d",
                                 "elixir_navierstokes_lid_driven_cavity.jl"),
                        cells_per_dimension=(4, 4), tspan=(0.0, 0.5),
                        l2=[
                            0.00022156125227115747,
                            0.028318325921401,
                            0.009509168701070296,
                            0.028267900513550506
                        ],
                        linf=[
                            0.001562278941298234,
                            0.14886653390744856,
                            0.0716323565533752,
                            0.19472785105241996
                        ])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "TreeMesh2D: elixir_advection_diffusion.jl" begin
    @test_trixi_include(joinpath(examples_dir(), "tree_2d_dgsem",
                                 "elixir_advection_diffusion.jl"),
                        initial_refinement_level=2, tspan=(0.0, 0.4), polydeg=5,
                        l2=[4.0915532997994255e-6],
                        linf=[2.3040850347877395e-5])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "TreeMesh2D: elixir_advection_diffusion.jl (Refined mesh)" begin
    @test_trixi_include(joinpath(examples_dir(), "tree_2d_dgsem",
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
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 100
        @test (@allocated Trixi.rhs_parabolic!(du_ode, u_ode, semi, t)) < 100
    end
end

@trixi_testset "TreeMesh2D: elixir_advection_diffusion_nonperiodic.jl" begin
    @test_trixi_include(joinpath(examples_dir(), "tree_2d_dgsem",
                                 "elixir_advection_diffusion_nonperiodic.jl"),
                        initial_refinement_level=2, tspan=(0.0, 0.1),
                        l2=[0.007646800618485118],
                        linf=[0.10067621050468958])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "TreeMesh2D: elixir_navierstokes_convergence.jl" begin
    @test_trixi_include(joinpath(examples_dir(), "tree_2d_dgsem",
                                 "elixir_navierstokes_convergence.jl"),
                        initial_refinement_level=2, tspan=(0.0, 0.1),
                        analysis_callback=AnalysisCallback(semi,
                                                           interval = analysis_interval,
                                                           extra_analysis_integrals = (energy_kinetic,
                                                                                       energy_internal,
                                                                                       enstrophy)),
                        l2=[
                            0.002111672530658797,
                            0.0034322351490857846,
                            0.0038742528195910416,
                            0.012469246082568561
                        ],
                        linf=[
                            0.012006418939223495,
                            0.035520871209746126,
                            0.024512747492231427,
                            0.11191122588756564
                        ])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "TreeMesh2D: elixir_navierstokes_convergence.jl (isothermal walls)" begin
    @test_trixi_include(joinpath(examples_dir(), "tree_2d_dgsem",
                                 "elixir_navierstokes_convergence.jl"),
                        initial_refinement_level=2, tspan=(0.0, 0.1),
                        heat_bc_top_bottom=Isothermal((x, t, equations) -> Trixi.temperature(initial_condition_navier_stokes_convergence_test(x,
                                                                                                                                              t,
                                                                                                                                              equations),
                                                                                             equations)),
                        l2=[
                            0.002103629650383915,
                            0.003435843933396454,
                            0.00386735987813341,
                            0.012670355349235728
                        ],
                        linf=[
                            0.012006261793147788,
                            0.03550212518982032,
                            0.025107947319661185,
                            0.11647078036571124
                        ])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "TreeMesh2D: elixir_navierstokes_convergence.jl (Entropy gradient variables)" begin
    @test_trixi_include(joinpath(examples_dir(), "tree_2d_dgsem",
                                 "elixir_navierstokes_convergence.jl"),
                        initial_refinement_level=2, tspan=(0.0, 0.1),
                        gradient_variables=GradientVariablesEntropy(),
                        l2=[
                            0.0021403742517389513,
                            0.0034258287094908572,
                            0.0038915122886898517,
                            0.012506862343013842
                        ],
                        linf=[
                            0.012244412004628336,
                            0.03507559186162224,
                            0.024580892345558894,
                            0.11425600758350107
                        ])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "TreeMesh2D: elixir_navierstokes_convergence.jl (Entropy gradient variables, isothermal walls)" begin
    @test_trixi_include(joinpath(examples_dir(), "tree_2d_dgsem",
                                 "elixir_navierstokes_convergence.jl"),
                        initial_refinement_level=2, tspan=(0.0, 0.1),
                        gradient_variables=GradientVariablesEntropy(),
                        heat_bc_top_bottom=Isothermal((x, t, equations) -> Trixi.temperature(initial_condition_navier_stokes_convergence_test(x,
                                                                                                                                              t,
                                                                                                                                              equations),
                                                                                             equations)),
                        l2=[
                            0.0021349737347844907,
                            0.0034301388278203033,
                            0.0038928324474291572,
                            0.012693611436230873
                        ],
                        linf=[
                            0.01224423627586213,
                            0.035054066314102905,
                            0.025099598504931965,
                            0.11795616324751634
                        ])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "TreeMesh2D: elixir_navierstokes_convergence.jl (flux differencing)" begin
    @test_trixi_include(joinpath(examples_dir(), "tree_2d_dgsem",
                                 "elixir_navierstokes_convergence.jl"),
                        initial_refinement_level=2, tspan=(0.0, 0.1),
                        volume_integral=VolumeIntegralFluxDifferencing(flux_central),
                        l2=[
                            0.0021116725306633594,
                            0.0034322351490827557,
                            0.0038742528196093542,
                            0.012469246082526909
                        ],
                        linf=[
                            0.012006418939291663,
                            0.035520871209594115,
                            0.024512747491801577,
                            0.11191122588591007
                        ])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "TreeMesh2D: elixir_navierstokes_convergence.jl (Refined mesh)" begin
    @test_trixi_include(joinpath(examples_dir(), "tree_2d_dgsem",
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
          [0.00024296959173852447; 0.0002093263158670915; 0.0005390572390977262;
           0.00026753561392341537]
    @test linf_error ≈
          [0.0016210102053424436; 0.002593287648655501; 0.002953907343823712;
           0.002077119120180271]
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "TreeMesh2D: elixir_navierstokes_lid_driven_cavity.jl" begin
    @test_trixi_include(joinpath(examples_dir(), "tree_2d_dgsem",
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
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "TreeMesh2D: elixir_navierstokes_shearlayer_amr.jl" begin
    @test_trixi_include(joinpath(examples_dir(), "tree_2d_dgsem",
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
    @test_trixi_include(joinpath(examples_dir(), "tree_2d_dgsem",
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
    @test_trixi_include(joinpath(examples_dir(), "p4est_2d_dgsem",
                                 "elixir_advection_diffusion_periodic.jl"),
                        trees_per_dimension=(1, 1), initial_refinement_level=2,
                        tspan=(0.0, 0.5),
                        l2=[0.0023754695605828443],
                        linf=[0.008154128363741964])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "P4estMesh2D: elixir_advection_diffusion_periodic.jl" begin
    @test_trixi_include(joinpath(examples_dir(), "p4est_2d_dgsem",
                                 "elixir_advection_diffusion_periodic.jl"),
                        trees_per_dimension=(1, 1), initial_refinement_level=2,
                        tspan=(0.0, 0.5),
                        l2=[0.0023754695605828443],
                        linf=[0.008154128363741964])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "P4estMesh2D: elixir_advection_diffusion_periodic_curved.jl" begin
    @test_trixi_include(joinpath(examples_dir(), "p4est_2d_dgsem",
                                 "elixir_advection_diffusion_periodic_curved.jl"),
                        trees_per_dimension=(1, 1), initial_refinement_level=2,
                        tspan=(0.0, 0.5),
                        l2=[0.006708147442490916],
                        linf=[0.04807038397976693])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "P4estMesh2D: elixir_advection_diffusion_periodic_amr.jl" begin
    @test_trixi_include(joinpath(examples_dir(), "p4est_2d_dgsem",
                                 "elixir_advection_diffusion_periodic_amr.jl"),
                        tspan=(0.0, 0.01),
                        l2=[0.014715887539773128],
                        linf=[0.2285802791900049])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "P4estMesh2D: elixir_advection_diffusion_nonperiodic_amr.jl" begin
    @test_trixi_include(joinpath(examples_dir(), "p4est_2d_dgsem",
                                 "elixir_advection_diffusion_nonperiodic_amr.jl"),
                        tspan=(0.0, 0.01),
                        l2=[0.007934195641974433],
                        linf=[0.11030265194954081])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "P4estMesh2D: elixir_advection_diffusion_nonperiodic_curved.jl" begin
    @test_trixi_include(joinpath(examples_dir(), "p4est_2d_dgsem",
                                 "elixir_advection_diffusion_nonperiodic_curved.jl"),
                        trees_per_dimension=(1, 1), initial_refinement_level=2,
                        tspan=(0.0, 0.5),
                        l2=[0.00919917034843865],
                        linf=[0.14186297438393505])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "P4estMesh2D: elixir_navierstokes_convergence.jl" begin
    @test_trixi_include(joinpath(examples_dir(), "p4est_2d_dgsem",
                                 "elixir_navierstokes_convergence.jl"),
                        initial_refinement_level=1, tspan=(0.0, 0.2),
                        l2=[
                            0.0003811978985836709,
                            0.0005874314969169538,
                            0.0009142898787923481,
                            0.0011613918899727263
                        ],
                        linf=[
                            0.0021633623982135752,
                            0.009484348274135372,
                            0.004231572066492217,
                            0.011661660275365193
                        ])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "P4estMesh2D: elixir_navierstokes_convergence_nonperiodic.jl" begin
    @test_trixi_include(joinpath(examples_dir(), "p4est_2d_dgsem",
                                 "elixir_navierstokes_convergence_nonperiodic.jl"),
                        initial_refinement_level=1, tspan=(0.0, 0.2),
                        l2=[
                            0.00040364962558511795,
                            0.0005869762481506936,
                            0.00091488537427274,
                            0.0011984191566376762
                        ],
                        linf=[
                            0.0024993634941723464,
                            0.009487866203944725,
                            0.004505829506628117,
                            0.011634902776245681
                        ])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "P4estMesh2D: elixir_navierstokes_lid_driven_cavity.jl" begin
    @test_trixi_include(joinpath(examples_dir(), "p4est_2d_dgsem",
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
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "P4estMesh2D: elixir_navierstokes_lid_driven_cavity_amr.jl" begin
    @test_trixi_include(joinpath(examples_dir(), "p4est_2d_dgsem",
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
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "elixir_navierstokes_NACA0012airfoil_mach08.jl" begin
    @test_trixi_include(joinpath(examples_dir(), "p4est_2d_dgsem",
                                 "elixir_navierstokes_NACA0012airfoil_mach08.jl"),
                        l2=[0.000186486564226516,
                            0.0005076712323400374,
                            0.00038074588984354107,
                            0.002128177239782089],
                        linf=[0.5153387072802718,
                            1.199362305026636,
                            0.9077214424040279,
                            5.666071182328691], tspan=(0.0, 0.001),
                        initial_refinement_level=0,
                        # With the default `maxiters = 1` in coverage tests,
                        # there would be no time steps after the restart.
                        coverage_override=(maxiters = 10_000,))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end

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
end

# Clean up afterwards: delete Trixi.jl output directory
@test_nowarn isdir(outdir) && rm(outdir, recursive = true)

end # module
