module TestExamplesP4estMesh2D

using Test
using Trixi

include("test_trixi.jl")

EXAMPLES_DIR = joinpath(examples_dir(), "p4est_2d_dgsem")

# Start with a clean environment: remove Trixi.jl output directory if it exists
outdir = "out"
isdir(outdir) && rm(outdir, recursive = true)

@testset "P4estMesh2D" begin
#! format: noindent

@trixi_testset "elixir_advection_basic.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_basic.jl"),
                        # Expected errors are exactly the same as with TreeMesh!
                        l2=[8.311947673061856e-6],
                        linf=[6.627000273229378e-5])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "elixir_advection_nonconforming_flag.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR,
                                 "elixir_advection_nonconforming_flag.jl"),
                        l2=[3.198940059144588e-5],
                        linf=[0.00030636069494005547])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "elixir_advection_unstructured_flag.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_unstructured_flag.jl"),
                        l2=[0.0005379687442422346],
                        linf=[0.007438525029884735])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "elixir_advection_amr_solution_independent.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR,
                                 "elixir_advection_amr_solution_independent.jl"),
                        # Expected errors are exactly the same as with StructuredMesh!
                        l2=[4.949660644033807e-5],
                        linf=[0.0004867846262313763],
                        coverage_override=(maxiters = 6,))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "elixir_advection_amr_unstructured_flag.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR,
                                 "elixir_advection_amr_unstructured_flag.jl"),
                        l2=[0.0012766060609964525],
                        linf=[0.01750280631586159],
                        coverage_override=(maxiters = 6,))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "elixir_advection_restart.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_restart.jl"),
                        l2=[4.507575525876275e-6],
                        linf=[6.21489667023134e-5],
                        # With the default `maxiters = 1` in coverage tests,
                        # there would be no time steps after the restart.
                        coverage_override=(maxiters = 25,))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "elixir_advection_restart_amr.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_restart_amr.jl"),
                        l2=[2.869137983727866e-6],
                        linf=[3.8353423270964804e-5],
                        # With the default `maxiters = 1` in coverage tests,
                        # there would be no time steps after the restart.
                        coverage_override=(maxiters = 25,))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "elixir_euler_source_terms_nonconforming_unstructured_flag.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR,
                                 "elixir_euler_source_terms_nonconforming_unstructured_flag.jl"),
                        l2=[
                            0.0034516244508588046,
                            0.0023420334036925493,
                            0.0024261923964557187,
                            0.004731710454271893,
                        ],
                        linf=[
                            0.04155789011775046,
                            0.024772109862748914,
                            0.03759938693042297,
                            0.08039824959535657,
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

@trixi_testset "elixir_euler_free_stream.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_free_stream.jl"),
                        l2=[
                            2.063350241405049e-15,
                            1.8571016296925367e-14,
                            3.1769447886391905e-14,
                            1.4104095258528071e-14,
                        ],
                        linf=[1.9539925233402755e-14, 2e-12, 4.8e-12, 4e-12],
                        atol=2.0e-12,)
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "elixir_euler_shockcapturing_ec.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_shockcapturing_ec.jl"),
                        l2=[
                            9.53984675e-02,
                            1.05633455e-01,
                            1.05636158e-01,
                            3.50747237e-01,
                        ],
                        linf=[
                            2.94357464e-01,
                            4.07893014e-01,
                            3.97334516e-01,
                            1.08142520e+00,
                        ],
                        tspan=(0.0, 1.0))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "elixir_euler_shockcapturing_ec.jl (flux_chandrashekar)" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_shockcapturing_ec.jl"),
                        l2=[
                            0.09527896382082567,
                            0.10557894830184737,
                            0.10559379376154387,
                            0.3503791205165925,
                        ],
                        linf=[
                            0.2733486454092644,
                            0.3877283966722886,
                            0.38650482703821426,
                            1.0053712251056308,
                        ],
                        tspan=(0.0, 1.0),
                        volume_flux=flux_chandrashekar)
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "elixir_euler_sedov.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_sedov.jl"),
                        l2=[
                            3.76149952e-01,
                            2.46970327e-01,
                            2.46970327e-01,
                            1.28889042e+00,
                        ],
                        linf=[
                            1.22139001e+00,
                            1.17742626e+00,
                            1.17742626e+00,
                            6.20638482e+00,
                        ],
                        tspan=(0.0, 0.3))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "elixir_euler_sedov.jl with HLLC Flux" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_sedov.jl"),
                        l2=[
                            0.4229948321239887,
                            0.2559038337457483,
                            0.2559038337457484,
                            1.2990046683564136,
                        ],
                        linf=[
                            1.4989357969730492,
                            1.325456585141623,
                            1.3254565851416251,
                            6.331283015053501,
                        ],
                        surface_flux=flux_hllc,
                        tspan=(0.0, 0.3))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "elixir_euler_sedov.jl (HLLE)" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_sedov.jl"),
                        l2=[
                            0.40853279043747015,
                            0.25356771650524296,
                            0.2535677165052422,
                            1.2984601729572691,
                        ],
                        linf=[
                            1.3840909333784284,
                            1.3077772519086124,
                            1.3077772519086157,
                            6.298798630968632,
                        ],
                        surface_flux=flux_hlle,
                        tspan=(0.0, 0.3))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "elixir_euler_blast_wave_amr.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_blast_wave_amr.jl"),
                        l2=[
                            6.32183914e-01,
                            3.86914231e-01,
                            3.86869171e-01,
                            1.06575688e+00,
                        ],
                        linf=[
                            2.76020890e+00,
                            2.32659890e+00,
                            2.32580837e+00,
                            2.15778188e+00,
                        ],
                        tspan=(0.0, 0.3),
                        coverage_override=(maxiters = 6,))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "elixir_euler_wall_bc_amr.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_wall_bc_amr.jl"),
                        l2=[
                            0.020291447969983396,
                            0.017479614254319948,
                            0.011387644425613437,
                            0.0514420126021293,
                        ],
                        linf=[
                            0.3582779022370579,
                            0.32073537890751663,
                            0.221818049107692,
                            0.9209559420400415,
                        ],
                        tspan=(0.0, 0.15))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "elixir_euler_forward_step_amr.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_forward_step_amr.jl"),
                        l2=[
                            0.004194875320833303,
                            0.003785140699353966,
                            0.0013696609105790351,
                            0.03265268616046424,
                        ],
                        linf=[
                            2.0585399781442852,
                            2.213428805506876,
                            3.862362410419163,
                            17.75187237459251,
                        ],
                        tspan=(0.0, 0.0001),
                        rtol=1.0e-7,
                        skip_coverage=true)
    if @isdefined sol # Skipped in coverage run
        # Ensure that we do not have excessive memory allocations
        # (e.g., from type instabilities)
        let
            t = sol.t[end]
            u_ode = sol.u[end]
            du_ode = similar(u_ode)
            @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
        end
    end
end

@trixi_testset "elixir_euler_double_mach_amr.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_double_mach_amr.jl"),
                        l2=[
                            0.051359355290192046,
                            0.4266034859911273,
                            0.2438304855475594,
                            4.11487176105527,
                        ],
                        linf=[
                            6.902000373057003,
                            53.95714139820832,
                            24.241610279839758,
                            561.0630401858057,
                        ],
                        tspan=(0.0, 0.0001),
                        skip_coverage=true)
    if @isdefined sol # Skipped in coverage run
        # Ensure that we do not have excessive memory allocations
        # (e.g., from type instabilities)
        let
            t = sol.t[end]
            u_ode = sol.u[end]
            du_ode = similar(u_ode)
            @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
        end
    end
end

@trixi_testset "elixir_euler_supersonic_cylinder.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_supersonic_cylinder.jl"),
                        l2=[
                            0.026798021911954406,
                            0.05118546368109259,
                            0.03206703583774831,
                            0.19680026567208672,
                        ],
                        linf=[
                            3.653905721692421,
                            4.285035711361009,
                            6.8544353186357645,
                            31.748244912257533,
                        ],
                        tspan=(0.0, 0.001),
                        skip_coverage=true)
    if @isdefined sol # Skipped in coverage run
        # Ensure that we do not have excessive memory allocations
        # (e.g., from type instabilities)
        let
            t = sol.t[end]
            u_ode = sol.u[end]
            du_ode = similar(u_ode)
            @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
        end
    end
end

@trixi_testset "elixir_euler_NACA6412airfoil_mach2.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_NACA6412airfoil_mach2.jl"),
                        l2=[
                            0.19107654776276498, 0.3545913719444839,
                            0.18492730895077583, 0.817927213517244,
                        ],
                        linf=[
                            2.5397624311491946, 2.7075156425517917, 2.200980534211764,
                            9.031153939238115,
                        ],
                        tspan=(0.0, 0.1))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "elixir_eulergravity_convergence.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_eulergravity_convergence.jl"),
                        l2=[
                            0.00024871265138964204,
                            0.0003370077102132591,
                            0.0003370077102131964,
                            0.0007231525513793697,
                        ],
                        linf=[
                            0.0015813032944647087,
                            0.0020494288423820173,
                            0.0020494288423824614,
                            0.004793821195083758,
                        ],
                        tspan=(0.0, 0.1))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "elixir_shallowwater_source_terms.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_shallowwater_source_terms.jl"),
                        l2=[
                            9.168126407325352e-5,
                            0.0009795410115453788,
                            0.002546408320320785,
                            3.941189812642317e-6,
                        ],
                        linf=[
                            0.0009903782521019089,
                            0.0059752684687262025,
                            0.010941106525454103,
                            1.2129488214718265e-5,
                        ],
                        tspan=(0.0, 0.1))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "elixir_mhd_alfven_wave.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhd_alfven_wave.jl"),
                        l2=[1.0513414461545583e-5, 1.0517900957166411e-6,
                            1.0517900957304043e-6, 1.511816606372376e-6,
                            1.0443997728645063e-6, 7.879639064990798e-7,
                            7.879639065049896e-7, 1.0628631669056271e-6,
                            4.3382328912336153e-7],
                        linf=[4.255466285174592e-5, 1.0029706745823264e-5,
                            1.0029706747467781e-5, 1.2122265939010224e-5,
                            5.4791097160444835e-6, 5.18922042269665e-6,
                            5.189220422141538e-6, 9.552667261422676e-6,
                            1.4237578427628152e-6])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "elixir_mhd_rotor.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhd_rotor.jl"),
                        l2=[0.4552084651735862, 0.8918048264575757, 0.832471223081887,
                            0.0,
                            0.9801264164951583, 0.10475690769435382, 0.1555132409149897,
                            0.0,
                            2.0597079362763556e-5],
                        linf=[10.194181233788775, 18.25472397868819, 10.031307436191334,
                            0.0,
                            19.647239392277378, 1.3938810140985936, 1.8724965294853084,
                            0.0,
                            0.0016290067532561904],
                        tspan=(0.0, 0.02))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "elixir_linearizedeuler_gaussian_source.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR,
                                 "elixir_linearizedeuler_gaussian_source.jl"),
                        l2=[
                            0.006047938590548741,
                            0.0040953286019907035,
                            0.004222698522497298,
                            0.006269492499336128,
                        ],
                        linf=[
                            0.06386175207349379,
                            0.0378926444850457,
                            0.041759728067967065,
                            0.06430136016259067,
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

@trixi_testset "elixir_euler_subsonic_cylinder.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_subsonic_cylinder.jl"),
                        l2=[
                            0.00011914390523852561,
                            0.00010776028621724485,
                            6.139954358305467e-5,
                            0.0003067693731825959,
                        ],
                        linf=[
                            0.1653075586200805,
                            0.1868437275544909,
                            0.09772818519679008,
                            0.4311796171737692,
                        ], tspan=(0.0, 0.001))
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
    drag = Trixi.analyze(drag_coefficient, du, u, tspan[2], mesh, equations, solver,
                         semi.cache, semi)
    lift = Trixi.analyze(lift_coefficient, du, u, tspan[2], mesh, equations, solver,
                         semi.cache, semi)

    @test isapprox(lift, -6.501138753497174e-15, atol = 1e-13)
    @test isapprox(drag, 2.588589856781827, atol = 1e-13)
end

# Forces computation test in an AMR code
@trixi_testset "elixir_euler_NACA0012airfoil_mach085.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR,
                                 "elixir_euler_NACA0012airfoil_mach085.jl"),
                        l2=[
                            5.634402680811982e-7, 6.748066107517321e-6,
                            1.091879472416885e-5, 0.0006686372064029146,
                        ],
                        linf=[
                            0.0021456247890772823, 0.03957142889488085,
                            0.03832024233032798, 2.6628739573358495,
                        ],
                        amr_interval=1,
                        base_level=0, med_level=1, max_level=1,
                        tspan=(0.0, 0.0001),
                        adapt_initial_condition=false,
                        adapt_initial_condition_only_refine=false,
                        # With the default `maxiters = 1` in coverage tests,
                        # the values for `drag` and `lift` below would differ.
                        coverage_override=(maxiters = 100_000,))

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
    drag = Trixi.analyze(drag_coefficient, du, u, tspan[2], mesh, equations, solver,
                         semi.cache, semi)
    lift = Trixi.analyze(lift_coefficient, du, u, tspan[2], mesh, equations, solver,
                         semi.cache, semi)

    @test isapprox(lift, 0.029076443678087403, atol = 1e-13)
    @test isapprox(drag, 0.13564720009197903, atol = 1e-13)
end

@trixi_testset "elixir_euler_blast_wave_pure_fv.jl" begin
    @test_trixi_include(joinpath(pkgdir(Trixi, "examples", "tree_2d_dgsem"),
                                 "elixir_euler_blast_wave_pure_fv.jl"),
                        l2=[
                            0.39957047631960346,
                            0.21006912294983154,
                            0.21006903549932,
                            0.6280328163981136,
                        ],
                        linf=[
                            2.20417889887697,
                            1.5487238480003327,
                            1.5486788679247812,
                            2.4656795949035857,
                        ],
                        tspan=(0.0, 0.5),
                        # Let this test run longer to cover some lines in flux_hllc
                        coverage_override=(maxiters = 10^5, tspan = (0.0, 0.1)),
                        mesh=mesh = P4estMesh((64, 64), polydeg = 3,
                                              coordinates_min = (-2.0, -2.0),
                                              coordinates_max = (2.0, 2.0)))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end
end

# Clean up afterwards: delete Trixi.jl output directory
@test_nowarn rm(outdir, recursive = true)

end # module
