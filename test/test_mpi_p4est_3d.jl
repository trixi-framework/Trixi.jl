@testsnippet MPIP4estMesh3D begin
    EXAMPLES_DIR = joinpath(examples_dir(), "p4est_3d_dgsem")
end

@testitem "P4estMesh MPI 3D: elixir_advection_basic.jl" setup=[Setup, MPIP4estMesh3D] tags=[
    :mpi,
    :mpi_skip_windows
] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_basic.jl"),
                        # Expected errors are exactly the same as with TreeMesh!
                        l2=[0.00016263963870641478],
                        linf=[0.0014537194925779984])

    @testset "error-based step size control" begin
        mpi_isroot() && println("-"^100)
        mpi_isroot() &&
            println("elixir_advection_basic.jl with error-based step size control")

        # Use callbacks without stepsize_callback to test error-based step size control
        callbacks = CallbackSet(summary_callback, analysis_callback, save_restart,
                                save_solution)
        sol = solve(ode, RDPK3SpFSAL35(); abstol = 1.0e-4, reltol = 1.0e-4,
                    ode_default_options()..., callback = callbacks)
        summary_callback()
        errors = analysis_callback(sol)
        if mpi_isroot()
            @test errors.l2≈[0.00016800412839949264] rtol=1.0e-4
            @test errors.linf≈[0.0014548839020096516] rtol=1.0e-4
        end
    end

    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
end

@testitem "P4estMesh MPI 3D: elixir_advection_amr.jl" setup=[Setup, MPIP4estMesh3D] tags=[
    :mpi,
    :mpi_skip_windows
] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_amr.jl"),
                        # Expected errors are exactly the same as with TreeMesh!
                        l2=[9.773852895157622e-6],
                        linf=[0.0005853874124926162])

    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
end

@testitem "P4estMesh MPI 3D: elixir_advection_amr_unstructured_curved.jl" setup=[
    Setup,
    MPIP4estMesh3D
] tags=[:mpi, :mpi_skip_windows] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR,
                                 "elixir_advection_amr_unstructured_curved.jl"),
                        l2=[1.6163120948209677e-5],
                        linf=[0.0010572201890564834],
                        tspan=(0.0, 1.0))

    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
end

@testitem "P4estMesh MPI 3D: elixir_advection_restart.jl" setup=[Setup, MPIP4estMesh3D] tags=[
    :mpi,
    :mpi_skip_windows
] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_restart.jl"),
                        l2=[0.002590388934758452],
                        linf=[0.01840757696885409])

    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
end

@testitem "P4estMesh MPI 3D: elixir_advection_cubed_sphere.jl" setup=[Setup, MPIP4estMesh3D] tags=[
    :mpi,
    :mpi_skip_windows
] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_cubed_sphere.jl"),
                        l2=[0.002006918015656413],
                        linf=[0.027655117058380085])

    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
end

# Compressible Euler
@testitem "P4estMesh MPI 3D: elixir_euler_source_terms_nonconforming_unstructured_curved.jl" setup=[
    Setup,
    MPIP4estMesh3D
] tags=[:mpi, :mpi_skip_windows] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR,
                                 "elixir_euler_source_terms_nonconforming_unstructured_curved.jl"),
                        l2=[
                            4.070355207909268e-5,
                            4.4993257426833716e-5,
                            5.10588457841744e-5,
                            5.102840924036687e-5,
                            0.00019986264001630542
                        ],
                        linf=[
                            0.0016987332417202072,
                            0.003622956808262634,
                            0.002029576258317789,
                            0.0024206977281964193,
                            0.008526972236273522
                        ],
                        tspan=(0.0, 0.01))

    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
end

@testitem "P4estMesh MPI 3D: elixir_euler_source_terms_nonperiodic.jl" setup=[
    Setup,
    MPIP4estMesh3D
] tags=[:mpi, :mpi_skip_windows] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR,
                                 "elixir_euler_source_terms_nonperiodic.jl"),
                        l2=[
                            0.0015106060984283647,
                            0.0014733349038567685,
                            0.00147333490385685,
                            0.001473334903856929,
                            0.0028149479453087093
                        ],
                        linf=[
                            0.008070806335238156,
                            0.009007245083113125,
                            0.009007245083121784,
                            0.009007245083102688,
                            0.01562861968368434
                        ],
                        tspan=(0.0, 1.0))

    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
end

@testitem "P4estMesh MPI 3D: elixir_euler_ec.jl" setup=[Setup, MPIP4estMesh3D] tags=[
    :mpi,
    :mpi_skip_windows
] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_ec.jl"),
                        l2=[
                            0.010380390326164493,
                            0.006192950051354618,
                            0.005970674274073704,
                            0.005965831290564327,
                            0.02628875593094754
                        ],
                        linf=[
                            0.3326911600075694,
                            0.2824952141320467,
                            0.41401037398065543,
                            0.45574161423218573,
                            0.8099577682187109
                        ],
                        tspan=(0.0, 0.2))

    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
end

@testitem "P4estMesh MPI 3D: elixir_euler_source_terms_nonperiodic_hohqmesh.jl" setup=[
    Setup,
    MPIP4estMesh3D
] tags=[:mpi, :mpi_skip_windows] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR,
                                 "elixir_euler_source_terms_nonperiodic_hohqmesh.jl"),
                        l2=[
                            0.0042023406458005464,
                            0.004122532789279737,
                            0.0042448149597303616,
                            0.0036361316700401765,
                            0.007389845952982495
                        ],
                        linf=[
                            0.04530610539892499,
                            0.02765695110527666,
                            0.05670295599308606,
                            0.048396544302230504,
                            0.1154589758186293
                        ])

    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
end

@testitem "P4estMesh MPI 3D: elixir_mhd_alfven_wave_nonconforming.jl" setup=[
    Setup,
    MPIP4estMesh3D
] tags=[:mpi, :mpi_skip_windows] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR,
                                 "elixir_mhd_alfven_wave_nonconforming.jl"),
                        l2=[
                            0.00017971806092102662,
                            0.000613696751148527,
                            0.00022822690442387818,
                            0.0007226600407756885,
                            0.000716160600125271,
                            0.0006066793297079394,
                            0.00027683929523418267,
                            0.0007102147790412368,
                            0.00018359046756853925
                        ],
                        linf=[
                            0.0011497146209857867,
                            0.00438491817590235,
                            0.0012563090800809806,
                            0.0046327018891885036,
                            0.0031258953293384906,
                            0.004268429815377006,
                            0.001771589463303802,
                            0.005043147336756829,
                            0.0014542473645126233
                        ],
                        tspan=(0.0, 0.25))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
end

@testitem "P4estMesh MPI 3D: elixir_mhd_alfven_wave_nonconforming.jl (combined fluxes)" setup=[
    Setup,
    MPIP4estMesh3D
] tags=[:mpi, :mpi_skip_windows] begin
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
                            0.00017971806092102662,
                            0.000613696751148527,
                            0.00022822690442387818,
                            0.0007226600407756885,
                            0.000716160600125271,
                            0.0006066793297079394,
                            0.00027683929523418267,
                            0.0007102147790412368,
                            0.00018359046756853925
                        ],
                        linf=[
                            0.0011497146209857867,
                            0.00438491817590235,
                            0.0012563090800809806,
                            0.0046327018891885036,
                            0.0031258953293384906,
                            0.004268429815377006,
                            0.001771589463303802,
                            0.005043147336756829,
                            0.0014542473645126233
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
@testitem "P4estMesh MPI 3D: elixir_mhd_alfven_wave_nonconforming.jl (one tree)" setup=[
    Setup,
    MPIP4estMesh3D
] tags=[:mpi, :mpi_skip_windows] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR,
                                 "elixir_mhd_alfven_wave_nonconforming.jl"),
                        l2=[
                            0.0018813659978217593,
                            0.0069867431133096725,
                            0.003385269870510109,
                            0.009070456066069444,
                            0.007731991775513,
                            0.007439982216025335,
                            0.0031468222855312515,
                            0.009048498228569343,
                            0.003710853679325309
                        ],
                        linf=[
                            0.01147065114383694,
                            0.04315730092920409,
                            0.020282820009729217,
                            0.048219992142552244,
                            0.04237665692161752,
                            0.040931199621710435,
                            0.026343478049302926,
                            0.05554610420155568,
                            0.024535345998684434
                        ],
                        tspan=(0.0, 0.25), trees_per_dimension=(1, 1, 1))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
end

@testitem "P4estMesh MPI 3D: combine_conservative_and_nonconservative_fluxes" setup=[
    Setup,
    MPIP4estMesh3D
] tags=[:mpi, :mpi_skip_windows] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR,
                                 "elixir_mhd_alfven_wave_combined_fluxes_nonperiodic.jl"),
                        l2=[
                            0.00016680585980070648,
                            0.0005174249857687432,
                            0.00027338410141730245,
                            0.0006282227969118486,
                            0.0006628677315608075,
                            0.0005583953631900829,
                            0.0001810882912027033,
                            0.0006112889578653592,
                            0.0002367494422736597
                        ],
                        linf=[
                            0.0009684118857454793,
                            0.0032142734886018987,
                            0.0018068280081933244,
                            0.0030945203920695052,
                            0.0037003192368261217,
                            0.0031660143839563615,
                            0.0011578323492362763,
                            0.003133093207396298,
                            0.0008290351027181377
                        ],
                        tspan=(0.0, 0.1))

    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
end

@testitem "P4estMesh MPI 3D: elixir_advection_limiter_liu_zhang.jl" setup=[
    Setup,
    MPIP4estMesh3D
] tags=[:mpi, :mpi_skip_windows] begin
    @test_trixi_include(joinpath(examples_dir(), "tree_3d_dgsem",
                                 "elixir_advection_limiter_liu_zhang.jl"),
                        mesh=P4estMesh((4, 4, 4), polydeg = 3,
                                       coordinates_min = (-1.0, -1.0, -1.0),
                                       coordinates_max = (1.0, 1.0, 1.0),
                                       initial_refinement_level = 1,
                                       periodicity = true),
                        # loosen tolerance since limiters can be sensitive
                        l2=[0.3917747509079011],
                        linf=[1.3569114615249707],
                        atol=5e-3, rtol=1e-2)
    u = Trixi.wrap_array_native(sol.u[end], semi)
    # matches thresholds = (1e-3,) up to a tolerance
    @test minimum(u) > 1e-3 - Trixi.euler_arithmetic_tol(1e-3, 1e-3)
    @test length(global_limiter!.history_davis_yin_iterations) > 0
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
end

@testitem "P4estMesh MPI 3D: check_axes" setup=[Setup, MPIP4estMesh3D] tags=[:mpi] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR,
                                 "elixir_euler_source_terms_nonconforming_unstructured_curved.jl"),
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
