@testsnippet MPIP4estMesh2D begin
    EXAMPLES_DIR = joinpath(examples_dir(), "p4est_2d_dgsem")
end

@testitem "P4estMesh MPI 2D: elixir_advection_basic.jl" setup=[Setup, MPIP4estMesh2D] tags=[:mpi] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_basic.jl"),
                        # Expected errors are exactly the same as with TreeMesh!
                        l2=[8.311947673061856e-6],
                        linf=[6.627000273229378e-5])

    @testset "error-based step size control" begin
        mpi_isroot() && println("-"^100)
        mpi_isroot() &&
            println("elixir_advection_basic.jl with error-based step size control")

        # Use callbacks without stepsize_callback to test error-based step size control
        callbacks = CallbackSet(summary_callback, analysis_callback, save_solution)
        sol = solve(ode, RDPK3SpFSAL35(); abstol = 1.0e-4, reltol = 1.0e-4,
                    ode_default_options()..., callback = callbacks)
        summary_callback()
        errors = analysis_callback(sol)
        if mpi_isroot()
            @test errors.l2≈[3.3022040342579066e-5] rtol=1.0e-4
            @test errors.linf≈[0.00011787417954578494] rtol=1.0e-4
        end
    end

    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
end

@testitem "P4estMesh MPI 2D: elixir_advection_nonconforming_flag.jl" setup=[
    Setup,
    MPIP4estMesh2D
] tags=[:mpi] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR,
                                 "elixir_advection_nonconforming_flag.jl"),
                        l2=[3.198940059144588e-5],
                        linf=[0.00030636069494005547])

    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
end

@testitem "P4estMesh MPI 2D: elixir_advection_unstructured_flag.jl" setup=[
    Setup,
    MPIP4estMesh2D
] tags=[:mpi] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR,
                                 "elixir_advection_unstructured_flag.jl"),
                        l2=[0.0005379687442422346],
                        linf=[0.007438525029884735])

    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
end

@testitem "P4estMesh MPI 2D: elixir_advection_amr_solution_independent.jl" setup=[
    Setup,
    MPIP4estMesh2D
] tags=[:mpi] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR,
                                 "elixir_advection_amr_solution_independent.jl"),
                        # Expected errors are exactly the same as with TreeMesh!
                        l2=[4.949660644033807e-5],
                        linf=[0.0004867846262313763])

    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
end

@testitem "P4estMesh MPI 2D: elixir_advection_amr_unstructured_flag.jl" setup=[
    Setup,
    MPIP4estMesh2D
] tags=[:mpi] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR,
                                 "elixir_advection_amr_unstructured_flag.jl"),
                        l2=[0.0012808538770535593],
                        linf=[0.01752690016659812])

    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
end

@testitem "P4estMesh MPI 2D: elixir_advection_restart.jl" setup=[Setup, MPIP4estMesh2D] tags=[:mpi] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_restart.jl"),
                        l2=[4.507575525876275e-6],
                        linf=[6.21489667023134e-5])

    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
end

@testitem "P4estMesh MPI 2D: elixir_euler_source_terms_nonconforming_unstructured_flag.jl" setup=[
    Setup,
    MPIP4estMesh2D
] tags=[:mpi] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR,
                                 "elixir_euler_source_terms_nonconforming_unstructured_flag.jl"),
                        l2=[
                            0.0034516244508588046,
                            0.0023420334036925493,
                            0.0024261923964557187,
                            0.004731710454271893
                        ],
                        linf=[
                            0.04155789011775046,
                            0.024772109862748914,
                            0.03759938693042297,
                            0.08039824959535657
                        ])

    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
end

# Test MPI-parallel handling of .inp meshes generated by HOHQMesh
@testitem "P4estMesh MPI 2D: elixir_euler_wall_bc_amr.jl" setup=[Setup, MPIP4estMesh2D] tags=[:mpi] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_wall_bc_amr.jl"),
                        l2=[
                            0.020266970819461425,
                            0.01746740120890609,
                            0.011378393090609054,
                            0.05138965928352185
                        ],
                        linf=[
                            0.3593492062888952,
                            0.32077672777509403,
                            0.23600493584887167,
                            0.9291837711500472
                        ],
                        tspan=(0.0, 0.15))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
end

# Test MPI-parallel handling of .inp meshes NOT generated by HOHQMesh
@testitem "P4estMesh MPI 2D: elixir_euler_SD7003airfoil.jl" setup=[Setup, MPIP4estMesh2D] tags=[:mpi] begin
    using Trixi: SemidiscretizationHyperbolic, AnalysisCallback
    @test_trixi_include(joinpath(EXAMPLES_DIR,
                                 "elixir_navierstokes_SD7003airfoil.jl"),
                        semi=SemidiscretizationHyperbolic(mesh, equations,
                                                          initial_condition, solver;
                                                          boundary_conditions = boundary_conditions_hyp),
                        analysis_callback=AnalysisCallback(semi,
                                                           interval = analysis_interval,
                                                           output_directory = "out",
                                                           save_analysis = true),
                        l2=[
                            9.316117984455285e-5,
                            4.539266936628966e-5,
                            8.381576796590632e-5,
                            0.00023437941500203496
                        ],
                        linf=[
                            0.31274105032407307,
                            0.2793016762668701,
                            0.22256470161743136,
                            0.7906704256076251
                        ],
                        tspan=(0.0, 5e-3))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
end

@testitem "P4estMesh MPI 2D: elixir_mhd_alfven_wave_nonconforming.jl" setup=[
    Setup,
    MPIP4estMesh2D
] tags=[:mpi] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR,
                                 "elixir_mhd_alfven_wave_nonconforming.jl"),
                        l2=[
                            0.03218873485054795,
                            0.03610423752891947,
                            0.03565782684252114,
                            0.052901719016820566,
                            0.033630607933966244,
                            0.03881219835771868,
                            0.03870276636879577,
                            0.052910258142801335,
                            0.022494653302015406
                        ],
                        linf=[
                            0.1795723574939876,
                            0.14031106752246264,
                            0.13219291731458843,
                            0.21691023663331932,
                            0.11730522693720591,
                            0.14520580597087362,
                            0.1442596290277719,
                            0.2175373736058324,
                            0.07672851650342115
                        ],
                        tspan=(0.0, 0.25))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
end

@testitem "P4estMesh MPI 2D: elixir_mhd_alfven_wave_nonconforming.jl (combined fluxes)" setup=[
    Setup,
    MPIP4estMesh2D
] tags=[:mpi] begin
    # Load the fluxes combining conservative and nonconservative terms,
    # which are equivalent to the fluxes used in the elixir below
    trixi_include(@__MODULE__,
                  joinpath(EXAMPLES_DIR,
                           "elixir_mhd_alfven_wave_combined_fluxes_nonperiodic.jl"),
                  maxiters = 1)

    # Use the same reference values as for the separate fluxes
    @test_trixi_include(joinpath(EXAMPLES_DIR,
                                 "elixir_mhd_alfven_wave_nonconforming.jl"),
                        l2=[
                            0.03218873485054795,
                            0.03610423752891947,
                            0.03565782684252114,
                            0.052901719016820566,
                            0.033630607933966244,
                            0.03881219835771868,
                            0.03870276636879577,
                            0.052910258142801335,
                            0.022494653302015406
                        ],
                        linf=[
                            0.1795723574939876,
                            0.14031106752246264,
                            0.13219291731458843,
                            0.21691023663331932,
                            0.11730522693720591,
                            0.14520580597087362,
                            0.1442596290277719,
                            0.2175373736058324,
                            0.07672851650342115
                        ],
                        solver=DGSEM(polydeg = 3,
                                     surface_flux = flux_hlle_nonconservative_powell,
                                     volume_integral = VolumeIntegralFluxDifferencing(flux_hindenlang_gassner_nonconservative_powell)),
                        tspan=(0.0, 0.25))

    # Make sure that the combined fluxes are used at MPI mortars (on some rank)
    @test Trixi.MPI.Allreduce(Trixi.nmpimortars(semi.solver, semi.cache), +,
                              Trixi.mpi_comm()) > 0

    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
end

# Same test as above but with only one tree in the mesh
# We use it to test meshes with elements of different size in each partition
@testitem "P4estMesh MPI 2D: elixir_mhd_alfven_wave_nonconforming.jl (one tree)" setup=[
    Setup,
    MPIP4estMesh2D
] tags=[:mpi] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR,
                                 "elixir_mhd_alfven_wave_nonconforming.jl"),
                        l2=[
                            0.029186071051606317,
                            0.03419919116962617,
                            0.03387081915185563,
                            0.050148091491269156,
                            0.03143527761821676,
                            0.037018590925082896,
                            0.03676486437523097,
                            0.05022930600012062,
                            0.021985956825544373
                        ],
                        linf=[
                            0.10530480293677158,
                            0.1360807239653038,
                            0.13403393077798956,
                            0.24345997266532407,
                            0.10222883223057888,
                            0.16191059057573343,
                            0.16758503888669074,
                            0.2437609564860478,
                            0.09472414618753822
                        ],
                        tspan=(0.0, 0.25), trees_per_dimension=(1, 1))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
end

@testitem "P4estMesh MPI 2D: combine_conservative_and_nonconservative_fluxes" setup=[
    Setup,
    MPIP4estMesh2D
] tags=[:mpi] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR,
                                 "elixir_mhd_alfven_wave_combined_fluxes_nonperiodic.jl"),
                        l2=[
                            7.475384025293373e-5,
                            6.508061190792402e-5,
                            6.494699189562051e-5,
                            0.00010039859002209644,
                            9.533883559804826e-5,
                            7.594652450345962e-5,
                            7.59909314809263e-5,
                            9.723421960617654e-5,
                            5.28916829920854e-5
                        ],
                        linf=[
                            0.0003427579185462104,
                            0.0004998303440422944,
                            0.0004839939536228077,
                            0.0005180908760881175,
                            0.0004278890186212303,
                            0.0004136795102115398,
                            0.0004278210815759076,
                            0.0005252948274237917,
                            0.00015843034878756284
                        ],
                        tspan=(0.0, 0.1))

    # Make sure that the combined fluxes are used at MPI interfaces (on some rank)
    @test Trixi.MPI.Allreduce(Trixi.nmpiinterfaces(semi.solver, semi.cache), +,
                              Trixi.mpi_comm()) > 0

    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
end

@testitem "P4estMesh MPI 2D: check_axes" setup=[Setup, MPIP4estMesh2D] tags=[:mpi] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR,
                                 "elixir_advection_nonconforming_flag.jl"),
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
