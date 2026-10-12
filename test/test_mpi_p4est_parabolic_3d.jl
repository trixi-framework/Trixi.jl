@testsnippet MPIP4estMesh3DParabolic begin
    EXAMPLES_DIR = joinpath(examples_dir(), "p4est_3d_dgsem")
end

@testitem "P4estMesh MPI 3D Parabolic: elixir_navierstokes_taylor_green_vortex.jl" setup=[
    Setup,
    MPIP4estMesh3DParabolic
] tags=[:mpi, :mpi_skip_windows] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR,
                                 "elixir_navierstokes_taylor_green_vortex.jl"),
                        initial_refinement_level=2, tspan=(0.0, 0.25),
                        surface_flux=FluxHLL(min_max_speed_naive),
                        l2=[
                            0.0001547509861140407,
                            0.015637861347119624,
                            0.015637861347119687,
                            0.022024699158522523,
                            0.009711013505930812
                        ],
                        linf=[
                            0.0006696415247340326,
                            0.03442565722527785,
                            0.03442565722577423,
                            0.06295407168705314,
                            0.032857472756916195
                        ])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1500)
    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1500)
end

@testitem "P4estMesh MPI 3D Parabolic: elixir_navierstokes_freestream_boundaries.jl" setup=[
    Setup,
    MPIP4estMesh3DParabolic
] tags=[:mpi, :mpi_skip_windows] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR,
                                 "elixir_navierstokes_freestream_boundaries.jl"),
                        tspan=(0.0, 0.1),
                        l2=[
                            1.050376383380673e-16,
                            1.0175313793753473e-16,
                            1.158489273890016e-16,
                            2.0654608507933775e-16,
                            3.3590256030698164e-15
                        ],
                        linf=[
                            1.7763568394002505e-15,
                            1.0130785099704553e-15,
                            1.3322676295501878e-15,
                            2.4424906541753444e-15,
                            4.263256414560601e-14
                        ])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1500)
    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1500)
end

@testitem "P4estMesh MPI 3D Parabolic: elixir_advection_diffusion_nonperiodic.jl (LDG)" setup=[
    Setup,
    MPIP4estMesh3DParabolic
] tags=[:mpi, :mpi_skip_windows] begin
    # The LDG interface fluxes depend on the direction of the normal vector. Thus,
    # both sides of an MPI interface must use the same (primary) normal direction
    # to obtain the same results as in serial. Moreover, the parabolic time step
    # restriction (`cfl_parabolic`) must be identical on all MPI ranks.
    @test_trixi_include(joinpath(EXAMPLES_DIR,
                                 "elixir_advection_diffusion_nonperiodic.jl"),
                        solver_parabolic=ParabolicFormulationLocalDG(),
                        cfl_parabolic=0.07,
                        l2=[0.004185076476662267], linf=[0.05166349548111486])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1500)
    @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1500)
end

@testitem "P4estMesh MPI 3D Parabolic: parabolic time step" setup=[
    Setup,
    MPIP4estMesh3DParabolic
] tags=[:mpi, :mpi_skip_windows] begin
    # The parabolic time step restriction must be identical on all MPI ranks
    # and coincide with the one of a serial simulation.
    function max_dt_parabolic(semi, ode)
        mesh, equations, solver, cache = Trixi.mesh_equations_solver_cache(semi)
        (; equations_parabolic) = semi
        u = Trixi.wrap_array(ode.u0, semi)
        return Trixi.max_dt(u, first(ode.tspan), mesh,
                            Trixi.have_constant_diffusivity(equations_parabolic),
                            equations, equations_parabolic, solver, cache)
    end
    is_identical_on_all_ranks(x) = Trixi.MPI.Allreduce(x, min, Trixi.mpi_comm()) ==
                                   Trixi.MPI.Allreduce(x, max, Trixi.mpi_comm())

    # Constant diffusivity on a curved mesh
    @test_trixi_include(joinpath(EXAMPLES_DIR,
                                 "elixir_advection_diffusion_nonperiodic.jl"),
                        tspan=(0.0, 0.0))
    dt = max_dt_parabolic(semi, ode)
    @test is_identical_on_all_ranks(dt)
    @test isapprox(dt, 0.015500439682961184, rtol = 1.0e-12)

    # Solution-dependent diffusivity
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_navierstokes_convergence.jl"),
                        initial_refinement_level=1, tspan=(0.0, 0.0))
    dt = max_dt_parabolic(semi, ode)
    @test is_identical_on_all_ranks(dt)
    @test isapprox(dt, 1.6071428571428439, rtol = 1.0e-12)
end
