@testsnippet MPITreeMesh2D begin
    EXAMPLES_DIR = joinpath(examples_dir(), "tree_2d_dgsem")
end

@testitem "TreeMesh MPI 2D: elixir_advection_basic.jl" setup=[Setup, MPITreeMesh2D] tags=[:mpi] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_basic.jl"),
                        # Expected errors are exactly the same as in the serial test!
                        l2=[8.311947673061856e-6],
                        linf=[6.627000273229378e-5])
end

@testitem "TreeMesh MPI 2D: elixir_advection_restart.jl" setup=[Setup, MPITreeMesh2D] tags=[:mpi] begin
    # Perform a standard simulation
    using OrdinaryDiffEqLowStorageRK: RDPK3SpFSAL49
    mpi_isroot() && println("═"^100)
    mpi_isroot() &&
        println(joinpath(EXAMPLES_DIR,
                         "elixir_advection_timeintegration_adaptive.jl"))
    trixi_include(@__MODULE__,
                  joinpath(EXAMPLES_DIR,
                           "elixir_advection_timeintegration_adaptive.jl"),
                  alg = RDPK3SpFSAL49(), tspan = (0.0, 10.0))
    l2_expected, linf_expected = analysis_callback(sol)

    # Perform a simulation restarting from an intermediate state
    mpi_isroot() && println("═"^100)
    mpi_isroot() &&
        println(joinpath(EXAMPLES_DIR, "elixir_advection_restart.jl"))
    trixi_include(@__MODULE__,
                  joinpath(EXAMPLES_DIR, "elixir_advection_restart.jl"),
                  alg = RDPK3SpFSAL49(),
                  base_elixir = "elixir_advection_timeintegration_adaptive.jl")
    l2_actual, linf_actual = analysis_callback(sol)

    # Check whether the errors are exactly the same as in the uninterrupted run
    # using the same low-storage RK method with error-based step size control.
    mpi_isroot() && @test l2_actual == l2_expected
    mpi_isroot() && @test linf_actual == linf_expected
end

@testitem "TreeMesh MPI 2D: elixir_advection_mortar.jl" setup=[Setup, MPITreeMesh2D] tags=[:mpi] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_mortar.jl"),
                        # Expected errors are exactly the same as in the serial test!
                        l2=[0.0015188466707237375],
                        linf=[0.008446655719187679])
end

@testitem "TreeMesh MPI 2D: check_axes" setup=[Setup, MPITreeMesh2D] tags=[:mpi] begin
    # Refine the upper half of the domain to get MPI mortars for 2 and 3 MPI ranks
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_mortar.jl"),
                        refinement_patches=((type = "box",
                                             coordinates_min = (-1.0, 0.0),
                                             coordinates_max = (1.0, 1.0)),),
                        maxiters=1)
    mesh, equations, dg, cache = Trixi.mesh_equations_solver_cache(semi)
    u = Trixi.wrap_array(Trixi.compute_coefficients(0.0, semi), semi)
    @test Trixi.check_axes(u, mesh, equations, dg, cache) === nothing

    # Make sure that there are MPI interfaces and MPI mortars (on some rank)
    @test Trixi.MPI.Allreduce(Trixi.nmpiinterfaces(dg, cache), +,
                              Trixi.mpi_comm()) > 0
    @test Trixi.MPI.Allreduce(Trixi.nmpimortars(dg, cache), +, Trixi.mpi_comm()) > 0
    for container in (cache.elements, cache.interfaces, cache.mpi_interfaces,
                      cache.boundaries, cache.mortars, cache.mpi_mortars)
        @test Trixi.check_axes(container, equations, dg, cache) === nothing
    end
end

@testitem "TreeMesh MPI 2D: linf_divb" setup=[Setup, MPITreeMesh2D] tags=[:mpi] begin
    # The maximum 8 of |div B| = |3 x^2 + 2 y + 3 y^2| is attained only at y = 1,
    # i.e., on some but not all MPI ranks. Polynomial degree 3 differentiates B exactly.
    equations = IdealGlmMhdEquations2D(1.4)
    function initial_condition(x, t, equations)
        B1 = x[1]^3
        B2 = x[2]^2 * (1 + x[2])
        prim = SVector(1.0, 0.0, 0.0, 0.0, 1.0, B1, B2, 0.0, 0.0)
        return prim2cons(prim, equations)
    end
    solver = DGSEM(polydeg = 3,
                   surface_flux = (flux_lax_friedrichs, flux_nonconservative_powell),
                   volume_integral = VolumeIntegralFluxDifferencing((flux_central,
                                                                     flux_nonconservative_powell)))
    mesh = TreeMesh((-1.0, -1.0), (1.0, 1.0), initial_refinement_level = 3,
                    periodicity = true)
    semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver;
                                        boundary_conditions = boundary_condition_periodic)
    u_ode = compute_coefficients(0.0, semi)
    u = Trixi.wrap_array(u_ode, semi)
    @test Trixi.analyze(Val(:linf_divb), nothing, u, 0.0, semi) ≈ 8
end

@testitem "TreeMesh MPI 2D: elixir_advection_amr.jl" setup=[Setup, MPITreeMesh2D] tags=[:mpi] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_amr.jl"),
                        # Expected errors are exactly the same as in the serial test!
                        l2=[4.913300828257469e-5],
                        linf=[0.00045263895394385967])
end

@testitem "TreeMesh MPI 2D: elixir_advection_amr_nonperiodic.jl" setup=[
    Setup,
    MPITreeMesh2D
] tags=[:mpi] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR,
                                 "elixir_advection_amr_nonperiodic.jl"),
                        # Expected errors are exactly the same as in the serial test!
                        l2=[3.2207388565869075e-5],
                        linf=[0.0007508059772436404])
end

@testitem "TreeMesh MPI 2D: elixir_advection_restart_amr.jl" setup=[Setup, MPITreeMesh2D] tags=[:mpi] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR,
                                 "elixir_advection_restart_amr.jl"),
                        l2=[8.018498574373939e-5],
                        linf=[0.0007307237754662355])
end

# Linear scalar advection with AMR
# These example files are only for testing purposes and have no practical use
@testitem "TreeMesh MPI 2D: elixir_advection_amr_refine_twice.jl" setup=[
    Setup,
    MPITreeMesh2D
] tags=[:mpi] begin
    using Trixi: Trixi
    # Here, we also test that SaveSolutionCallback prints multiple mesh files with AMR
    # Start with a clean environment: remove Trixi.jl output directory if it exists
    outdir = "out"
    mpi_isroot() && isdir(outdir) && rm(outdir, recursive = true)
    Trixi.MPI.Barrier(Trixi.mpi_comm())
    @test_trixi_include(joinpath(EXAMPLES_DIR,
                                 "elixir_advection_amr_refine_twice.jl"),
                        l2=[0.00020547512522578292],
                        linf=[0.007831753383083506])
    meshfiles = filter(file -> endswith(file, ".h5") && startswith(file, "mesh"),
                       readdir(outdir))
    @test length(meshfiles) > 1
end

@testitem "TreeMesh MPI 2D: elixir_advection_amr_coarsen_twice.jl" setup=[
    Setup,
    MPITreeMesh2D
] tags=[:mpi] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR,
                                 "elixir_advection_amr_coarsen_twice.jl"),
                        l2=[0.0014321062757891826],
                        linf=[0.0253454486893413])
end

# Hyperbolic diffusion
@testitem "TreeMesh MPI 2D: elixir_hypdiff_lax_friedrichs.jl" setup=[Setup, MPITreeMesh2D] tags=[
    :mpi,
    :mpi_skip_windows
] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR,
                                 "elixir_hypdiff_lax_friedrichs.jl"),
                        l2=[
                            0.00015687751816056159,
                            0.001025986772217084,
                            0.0010259867722169909
                        ],
                        linf=[
                            0.0011986956416591976,
                            0.006423873516411049,
                            0.006423873516411049
                        ])
end

@testitem "TreeMesh MPI 2D: elixir_hypdiff_harmonic_nonperiodic.jl" setup=[
    Setup,
    MPITreeMesh2D
] tags=[:mpi] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR,
                                 "elixir_hypdiff_harmonic_nonperiodic.jl"),
                        l2=[
                            8.61813235543625e-8,
                            5.619399844542781e-7,
                            5.6193998447443e-7
                        ],
                        linf=[
                            1.124861862180196e-6,
                            8.622436471039663e-6,
                            8.622436470151484e-6
                        ])
end

@testitem "TreeMesh MPI 2D: elixir_hypdiff_nonperiodic.jl" setup=[Setup, MPITreeMesh2D] tags=[:mpi] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_hypdiff_nonperiodic.jl"),
                        l2=[
                            8.523077653955306e-6,
                            2.8779323653065056e-5,
                            5.4549427691297846e-5
                        ],
                        linf=[
                            5.5227409524905013e-5,
                            0.0001454489597927185,
                            0.00032396328684569653
                        ])
end

@testitem "TreeMesh MPI 2D: elixir_hypdiff_godunov.jl" setup=[Setup, MPITreeMesh2D] tags=[
    :mpi,
    :mpi_skip_windows
] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_hypdiff_godunov.jl"),
                        l2=[
                            5.868147556427088e-6,
                            3.80517927324465e-5,
                            3.805179273249344e-5
                        ],
                        linf=[
                            3.701965498725812e-5,
                            0.0002122422943138247,
                            0.00021224229431116015
                        ],
                        atol=2.0e-12) #= required for CI on macOS =#
end

# Compressible Euler
# Note: Some tests here have manually increased relative tolerances since reduction via MPI can
#       slightly change the L2 error norms (different floating point truncation errors)
@testitem "TreeMesh MPI 2D: elixir_euler_source_terms.jl" setup=[Setup, MPITreeMesh2D] tags=[
    :mpi,
    :mpi_skip_windows
] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_source_terms.jl"),
                        l2=[
                            9.321181253186009e-7,
                            1.4181210743438511e-6,
                            1.4181210743487851e-6,
                            4.824553091276693e-6
                        ],
                        linf=[
                            9.577246529612893e-6,
                            1.1707525976012434e-5,
                            1.1707525976456523e-5,
                            4.8869615580926506e-5
                        ],
                        rtol=2000 * sqrt(eps()))
end

# This example file is only for testing purposes and has no practical use
@testitem "TreeMesh MPI 2D: elixir_euler_source_terms_amr_refine_coarsen.jl" setup=[
    Setup,
    MPITreeMesh2D
] tags=[:mpi, :mpi_skip_windows] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR,
                                 "elixir_euler_source_terms_amr_refine_coarsen.jl"),
                        l2=[
                            4.8226610349853444e-5,
                            4.117706709270575e-5,
                            4.1177067092959676e-5,
                            0.00012205252427437389
                        ],
                        linf=[
                            0.0003543874851490436,
                            0.0002973166773747593,
                            0.0002973166773760916,
                            0.001154106793870291
                        ])
end

@testitem "TreeMesh MPI 2D: elixir_euler_source_terms_nonperiodic.jl" setup=[
    Setup,
    MPITreeMesh2D
] tags=[:mpi, :mpi_skip_windows] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR,
                                 "elixir_euler_source_terms_nonperiodic.jl"),
                        l2=[
                            2.259440511766445e-6,
                            2.318888155713922e-6,
                            2.3188881557894307e-6,
                            6.3327863238858925e-6
                        ],
                        linf=[
                            1.498738264560373e-5,
                            1.9182011928187137e-5,
                            1.918201192685487e-5,
                            6.0526717141407005e-5
                        ],
                        rtol=0.001)
end

@testitem "TreeMesh MPI 2D: elixir_euler_ec.jl" setup=[Setup, MPITreeMesh2D] tags=[
    :mpi,
    :mpi_skip_windows
] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_ec.jl"),
                        l2=[
                            0.061751715597716854,
                            0.05018223615408711,
                            0.05018989446443463,
                            0.225871559730513
                        ],
                        linf=[
                            0.29347582879608825,
                            0.31081249232844693,
                            0.3107380389947736,
                            1.0540358049885143
                        ])

    @testset "error-based step size control" begin
        mpi_isroot() && println("-"^100)
        mpi_isroot() &&
            println("elixir_euler_ec.jl with error-based step size control")

        # Use callbacks without stepsize_callback to test error-based step size control
        callbacks = CallbackSet(summary_callback,
                                analysis_callback, alive_callback,
                                save_solution)
        sol = solve(ode, RDPK3SpFSAL35(); abstol = 1.0e-4, reltol = 1.0e-4,
                    ode_default_options()..., callback = callbacks)
        summary_callback()
        errors = analysis_callback(sol)
        if mpi_isroot()
            @test errors.l2≈[
                0.061653630426688116,
                0.05006930431098764,
                0.05007694316484242,
                0.22550689872331683
            ] rtol=1.0e-4
            @test errors.linf≈[
                0.28516937484583693,
                0.2983633696512788,
                0.297812036335975,
                1.027368795517512
            ] rtol=1.0e-4
        end
    end
end

@testitem "TreeMesh MPI 2D: elixir_euler_vortex.jl" setup=[Setup, MPITreeMesh2D] tags=[:mpi] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_vortex.jl"),
                        l2=[
                            0.00013492249515826863,
                            0.006615696236378061,
                            0.006782108219800376,
                            0.016393831451740604
                        ],
                        linf=[
                            0.0020782600954247776,
                            0.08150078921935999,
                            0.08663621974991986,
                            0.2829930622010579
                        ],
                        rtol=0.001)
end

@testitem "TreeMesh MPI 2D: elixir_euler_vortex_mortar.jl" setup=[Setup, MPITreeMesh2D] tags=[:mpi] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_vortex_mortar.jl"),
                        # Expected errors are exactly the same as in the serial test!
                        l2=[
                            3.1363505551305216e-5,
                            0.0006614564510650079,
                            0.0006466955139840528,
                            0.002661217863027477
                        ],
                        linf=[
                            0.0010628052760547346,
                            0.028186424944457555,
                            0.01130123802781463,
                            0.07516351234122709
                        ])
end

@testitem "TreeMesh MPI 2D: elixir_euler_vortex_amr.jl" setup=[Setup, MPITreeMesh2D] tags=[:mpi] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_vortex_amr.jl"),
                        # Expected errors are exactly the same as in the serial test!
                        l2=[
                            5.051719943432265e-5,
                            0.0022574259317084747,
                            0.0021755998463189713,
                            0.004346492398617521
                        ],
                        linf=[
                            0.0012880114865917447,
                            0.03857193149447702,
                            0.031090457959835893,
                            0.12125130332971423
                        ])
end

@testitem "TreeMesh MPI 2D: elixir_euler_vortex_shockcapturing.jl" setup=[
    Setup,
    MPITreeMesh2D
] tags=[:mpi, :mpi_skip_windows] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR,
                                 "elixir_euler_vortex_shockcapturing.jl"),
                        l2=[
                            0.0017158367642679273,
                            0.09619888722871434,
                            0.09616432767924141,
                            0.17553381166255197
                        ],
                        linf=[
                            0.021853862449723982,
                            0.9878047229255944,
                            0.9880191167111795,
                            2.2154030488035588
                        ],
                        rtol=0.001)
end

@testitem "TreeMesh MPI 2D: elixir_mhd_alfven_wave.jl" setup=[Setup, MPITreeMesh2D] tags=[:mpi] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhd_alfven_wave.jl"),
                        # Expected errors are exactly the same as in the serial test!
                        l2=[
                            0.00011149543672225127,
                            5.888242524520296e-6,
                            5.888242524510072e-6,
                            8.476931432519067e-6,
                            1.3160738644036652e-6,
                            1.2542675002588144e-6,
                            1.2542675002747718e-6,
                            1.8705223407238346e-6,
                            4.651717010670585e-7
                        ],
                        linf=[
                            0.00026806333988971254,
                            1.6278838272418272e-5,
                            1.627883827305665e-5,
                            2.7551183488072617e-5,
                            5.457878055614707e-6,
                            8.130129322880819e-6,
                            8.130129322769797e-6,
                            1.2406302192291552e-5,
                            2.373765544951732e-6
                        ])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
end

@testitem "TreeMesh MPI 2D: elixir_mhd_alfven_wave_mortar.jl" setup=[Setup, MPITreeMesh2D] tags=[:mpi] begin
    # Refine the upper half of the domain to get MPI mortars for 2 and 3 MPI ranks
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhd_alfven_wave_mortar.jl"),
                        refinement_patches=((type = "box",
                                             coordinates_min = (0.0, 0.5 * sqrt(2.0)),
                                             coordinates_max = (sqrt(2.0), sqrt(2.0))),),
                        # Expected errors are exactly the same as in a serial run!
                        l2=[
                            1.6786707558867175e-6,
                            7.164959462053942e-7,
                            7.081932438687196e-7,
                            1.0823779092603235e-6,
                            8.54211716322804e-7,
                            7.991340679276956e-7,
                            7.919017392071279e-7,
                            1.1361442054079598e-6,
                            3.9639297967084265e-7
                        ],
                        linf=[
                            2.9733402099330064e-5,
                            9.008923548395331e-6,
                            8.587065454285225e-6,
                            1.2054960366242318e-5,
                            7.867653680326825e-6,
                            8.938923408141619e-6,
                            8.920966544256892e-6,
                            1.2062249465857877e-5,
                            4.619996814003757e-6
                        ],
                        tspan=(0.0, 1.0))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)

    # Make sure that there are MPI interfaces and MPI mortars (on some rank)
    mesh, equations, dg, cache = Trixi.mesh_equations_solver_cache(semi)
    @test Trixi.MPI.Allreduce(Trixi.nmpiinterfaces(dg, cache), +,
                              Trixi.mpi_comm()) > 0
    @test Trixi.MPI.Allreduce(Trixi.nmpimortars(dg, cache), +, Trixi.mpi_comm()) > 0
end

@testitem "TreeMesh MPI 2D: elixir_advection_limiter_liu_zhang.jl" setup=[
    Setup,
    MPITreeMesh2D
] tags=[:mpi] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR,
                                 "elixir_advection_limiter_liu_zhang.jl"),
                        # loosen tolerance since limiters can be sensitive
                        l2=[0.204437455578503], linf=[0.8175130986098857],
                        atol=5e-3, rtol=1e-2,
                        record_davis_yin_iterations=true)
    u = Trixi.wrap_array_native(sol.u[end], semi)
    @test minimum(u) > 1e-1 - 10 * eps()
    @test length(global_limiter!.history_davis_yin_iterations) > 0
end
