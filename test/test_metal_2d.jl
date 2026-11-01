# Test items tagged `:metal_skip_m1_m2` fail on Apple M1/M2 GPUs because of upstream
# bugs in Metal.jl and are skipped there, see `runtests.jl`.

@testsnippet Metal2DExamples begin
    EXAMPLES_DIR = joinpath(examples_dir(), "p4est_2d_dgsem")
end

@testitem "Metal 2D: elixir_advection_basic.jl Float32 / Metal" setup=[
    Setup,
    Metal2DExamples
] tags=[:Metal] begin
    # Using Metal inside the testitem since otherwise the bindings are hidden by the anonymous modules
    using Metal
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_basic.jl"),
                        # Expected errors are exactly the same as with TreeMesh!
                        l2=[Float32(8.311947673061856e-6)],
                        linf=[Float32(6.627000273229378e-5)],
                        RealT_for_test_tolerances=Float32,
                        real_type=Float32,
                        storage_type=MtlArray)
    # The elixir adapts `semi` before creating the ODE problem
    @test semi === ode.p
    @test real(semi.solver) == Float32
    @test real(semi.solver.basis) == Float32
    @test real(semi.solver.mortar) == Float32
    # TODO: `mesh` is currently not `adapt`ed correctly
    @test real(semi.mesh) == Float64
    @test eltype(semi.equations.advection_velocity) == Float32

    @test ode.u0 isa MtlArray
    @test semi.solver.basis.derivative_matrix isa MtlArray

    @test Trixi.storage_type(semi.cache.elements) === MtlArray
    @test Trixi.storage_type(semi.cache.interfaces) === MtlArray
    @test Trixi.storage_type(semi.cache.boundaries) === MtlArray
    @test Trixi.storage_type(semi.cache.mortars) === MtlArray

    # Ensure that the RHS computation overwrites existing data in `du` correctly.
    u_ode = copy(ode.u0)
    du_ode = similar(u_ode)
    fill!(du_ode, convert(eltype(du_ode), NaN))
    Trixi.rhs_hyperbolic!(du_ode, u_ode, semi, first(ode.tspan))
    @test all(isfinite, du_ode)
end

@testitem "Metal 2D: elixir_advection_nonconforming_flag.jl Float32 / Metal" setup=[
    Setup,
    Metal2DExamples
] tags=[:Metal] begin
    # Using Metal inside the testitem since otherwise the bindings are hidden by the anonymous modules
    using Metal
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_nonconforming_flag.jl"),
                        # Expected errors similar to reference on CPU
                        l2=Float32[3.198940059144588e-5],
                        linf=Float32[0.00030636069494005547],
                        RealT_for_test_tolerances=Float32,
                        real_type=Float32,
                        storage_type=MtlArray)
    # The elixir adapts `semi` before creating the ODE problem
    @test semi === ode.p
    @test real(semi.solver) == Float32
    @test real(semi.solver.basis) == Float32
    @test real(semi.solver.mortar) == Float32
    # TODO: `mesh` is currently not `adapt`ed correctly
    @test real(semi.mesh) == Float64
    @test eltype(semi.equations.advection_velocity) == Float32

    # Make sure that the mortar kernels are used
    @test Trixi.nmortars(semi.solver, semi.cache) > 0
    @test ode.u0 isa MtlArray
    @test semi.solver.basis.derivative_matrix isa MtlArray

    @test Trixi.storage_type(semi.cache.elements) === MtlArray
    @test Trixi.storage_type(semi.cache.interfaces) === MtlArray
    @test Trixi.storage_type(semi.cache.boundaries) === MtlArray
    @test Trixi.storage_type(semi.cache.mortars) === MtlArray

    # Ensure that the RHS computation overwrites existing data in `du` correctly.
    u_ode = copy(ode.u0)
    du_ode = similar(u_ode)
    fill!(du_ode, convert(eltype(du_ode), NaN))
    Trixi.rhs_hyperbolic!(du_ode, u_ode, semi, first(ode.tspan))
    @test all(isfinite, du_ode)
end

@testitem "Metal 2D: elixir_euler_source_terms_nonconforming_unstructured_flag.jl Float32 / Metal" setup=[
    Setup,
    Metal2DExamples
] tags=[:Metal, :metal_skip_m1_m2] begin
    # Using Metal inside the testitem since otherwise the bindings are hidden by the anonymous modules
    using Metal
    @test_trixi_include(joinpath(EXAMPLES_DIR,
                                 "elixir_euler_source_terms_nonconforming_unstructured_flag.jl"),
                        # Expected errors similar to reference on CPU
                        l2=Float32[0.0034516244508588046,
                                   0.0023420334036925493,
                                   0.0024261923964557187,
                                   0.004731710454271893],
                        linf=Float32[0.04155789011775046,
                                     0.024772109862748914,
                                     0.03759938693042297,
                                     0.08039824959535657],
                        RealT_for_test_tolerances=Float32,
                        real_type=Float32,
                        storage_type=MtlArray)
    # The elixir adapts `semi` before creating the ODE problem
    @test semi === ode.p
    @test real(semi.solver) == Float32
    @test real(semi.solver.basis) == Float32
    @test real(semi.solver.mortar) == Float32
    # TODO: `mesh` is currently not `adapt`ed correctly
    @test real(semi.mesh) == Float64
    @test typeof(semi.equations.gamma) == Float32

    # Make sure that the mortar kernels are used, including mortars
    # whose large side runs backwards (flipped orientation)
    @test Trixi.nmortars(semi.solver, semi.cache) > 0
    @test any(indices -> NodeIndex.IBackward in indices,
              Array(semi.cache.mortars.node_indices)[2, :])
    @test ode.u0 isa MtlArray
    @test semi.solver.basis.derivative_matrix isa MtlArray

    @test Trixi.storage_type(semi.cache.elements) === MtlArray
    @test Trixi.storage_type(semi.cache.interfaces) === MtlArray
    @test Trixi.storage_type(semi.cache.boundaries) === MtlArray
    @test Trixi.storage_type(semi.cache.mortars) === MtlArray

    # Ensure that the RHS computation overwrites existing data in `du` correctly.
    u_ode = copy(ode.u0)
    du_ode = similar(u_ode)
    fill!(du_ode, convert(eltype(du_ode), NaN))
    Trixi.rhs_hyperbolic!(du_ode, u_ode, semi, first(ode.tspan))
    @test all(isfinite, du_ode)
end

@testitem "Metal 2D: elixir_euler_source_terms.jl Float32 / Metal" setup=[
    Setup,
    Metal2DExamples
] tags=[:Metal] begin
    # Using Metal inside the testitem since otherwise the bindings are hidden by the anonymous modules
    using Metal
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_source_terms.jl"),
                        l2=Float32[2.4917018095933837e-6,
                                   2.7148269885239423e-6,
                                   2.695290306860358e-6,
                                   6.243861976167833e-6],
                        linf=Float32[1.6489475493930428e-5,
                                     1.7499923706143505e-5,
                                     1.893043518075288e-5,
                                     6.214141845717336e-5],
                        RealT_for_test_tolerances=Float32,
                        real_type=Float32,
                        storage_type=MtlArray)
    # The elixir adapts `semi` before creating the ODE problem
    @test semi === ode.p
    @test real(semi.solver) == Float32
    @test real(semi.solver.basis) == Float32
    @test real(semi.solver.mortar) == Float32
    # TODO: `mesh` is currently not `adapt`ed correctly
    @test real(semi.mesh) == Float64
    @test typeof(semi.equations.gamma) == Float32

    @test ode.u0 isa MtlArray
    @test semi.solver.basis.derivative_matrix isa MtlArray

    @test Trixi.storage_type(semi.cache.elements) === MtlArray
    @test Trixi.storage_type(semi.cache.interfaces) === MtlArray
    @test Trixi.storage_type(semi.cache.boundaries) === MtlArray
    @test Trixi.storage_type(semi.cache.mortars) === MtlArray

    # Ensure that the RHS computation overwrites existing data in `du` correctly.
    u_ode = copy(ode.u0)
    du_ode = similar(u_ode)
    fill!(du_ode, convert(eltype(du_ode), NaN))
    Trixi.rhs_hyperbolic!(du_ode, u_ode, semi, first(ode.tspan))
    @test all(isfinite, du_ode)
end

@testitem "Metal 2D: elixir_euler_source_terms.jl Flux Differencing Float32 / Metal" setup=[
    Setup,
    Metal2DExamples
] tags=[:Metal] begin
    # Using Metal inside the testitem since otherwise the bindings are hidden by the anonymous modules
    using Metal
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_source_terms.jl"),
                        l2=Float32[2.7905685982444506e-6,
                                   2.7719663804722356e-6,
                                   2.862595247100584e-6,
                                   6.59779451858695e-6],
                        linf=Float32[1.904964447030366e-5,
                                     2.1734684234164803e-5,
                                     1.988410949715913e-5,
                                     5.9757232666157734e-5],
                        RealT_for_test_tolerances=Float32,
                        real_type=Float32,
                        storage_type=MtlArray,
                        solver=DGSEM(polydeg = 3,
                                     surface_flux = FluxLaxFriedrichs(max_abs_speed_naive),
                                     volume_integral = VolumeIntegralFluxDifferencing(flux_kennedy_gruber)))
    # The elixir adapts `semi` before creating the ODE problem
    @test semi === ode.p
    @test real(semi.solver) == Float32
    @test real(semi.solver.basis) == Float32
    @test real(semi.solver.mortar) == Float32
    # TODO: `mesh` is currently not `adapt`ed correctly
    @test real(semi.mesh) == Float64
    @test typeof(semi.equations.gamma) == Float32

    @test ode.u0 isa MtlArray
    @test semi.solver.basis.derivative_matrix isa MtlArray

    @test Trixi.storage_type(semi.cache.elements) === MtlArray
    @test Trixi.storage_type(semi.cache.interfaces) === MtlArray
    @test Trixi.storage_type(semi.cache.boundaries) === MtlArray
    @test Trixi.storage_type(semi.cache.mortars) === MtlArray

    # Ensure that the RHS computation overwrites existing data in `du` correctly.
    u_ode = copy(ode.u0)
    du_ode = similar(u_ode)
    fill!(du_ode, convert(eltype(du_ode), NaN))
    Trixi.rhs_hyperbolic!(du_ode, u_ode, semi, first(ode.tspan))
    @test all(isfinite, du_ode)
end

@testitem "Metal 2D: elixir_mhd_alfven_wave_combined_fluxes_nonperiodic.jl Float32 / Metal" setup=[
    Setup,
    Metal2DExamples
] tags=[:Metal, :metal_skip_m1_m2] begin
    # Using Metal inside the testitem since otherwise the bindings are hidden by the anonymous modules
    using Metal
    using Trixi
    @test_trixi_include(joinpath(EXAMPLES_DIR,
                                 "elixir_mhd_alfven_wave_combined_fluxes_nonperiodic.jl"),
                        l2=Float32[7.522484207987927e-5,
                                   6.320027873022108e-5,
                                   6.326290061132241e-5,
                                   0.00011731785274617709,
                                   7.481423726295819e-5,
                                   7.911056635500293e-5,
                                   7.935093823090327e-5,
                                   0.0001147135801732915,
                                   5.03544803621344e-5],
                        linf=Float32[0.0004317760467529297,
                                     0.0005172195608723826,
                                     0.0005215731631947051,
                                     0.0009205658752724083,
                                     0.0004047385522517599,
                                     0.0005733146458962723,
                                     0.0005431522323861016,
                                     0.0008944143373772057,
                                     0.0002989873755723238],
                        RealT_for_test_tolerances=Float32,
                        real_type=Float32,
                        storage_type=MtlArray)
    # The elixir adapts `semi` before creating the ODE problem
    @test semi === ode.p
    @test real(semi.solver) == Float32
    @test real(semi.solver.basis) == Float32
    @test real(semi.solver.mortar) == Float32
    # TODO: `mesh` is currently not `adapt`ed correctly
    @test real(semi.mesh) == Float64
    @test typeof(semi.equations.gamma) == Float32

    @test ode.u0 isa MtlArray
    @test semi.solver.basis.derivative_matrix isa MtlArray

    @test Trixi.storage_type(semi.cache.elements) === MtlArray
    @test Trixi.storage_type(semi.cache.interfaces) === MtlArray
    @test Trixi.storage_type(semi.cache.boundaries) === MtlArray
    @test Trixi.storage_type(semi.cache.mortars) === MtlArray

    # Ensure that the RHS computation overwrites existing data in `du` correctly.
    u_ode = copy(ode.u0)
    du_ode = similar(u_ode)
    fill!(du_ode, convert(eltype(du_ode), NaN))
    Trixi.rhs_hyperbolic!(du_ode, u_ode, semi, first(ode.tspan))
    @test all(isfinite, du_ode)
end

@testitem "Metal 2D: elixir_mhd_alfven_wave_combined_fluxes_nonperiodic.jl nonconforming Float32 / Metal" setup=[
    Setup,
    Metal2DExamples
] tags=[:Metal, :metal_skip_m1_m2] begin
    # Using Metal inside the testitem since otherwise the bindings are hidden by the anonymous modules
    using Metal
    using Trixi
    # Create a non-conforming mesh by refining the bottom left quadrant of each tree
    # to level 4 as in `elixir_mhd_alfven_wave_nonconforming.jl`, but without
    # periodicity to match the boundary conditions of the elixir used below
    mesh_nonconforming = P4estMesh((2, 2), polydeg = 3, initial_refinement_level = 2,
                                   coordinates_min = (-1.0, -1.0),
                                   coordinates_max = (1.0, 1.0),
                                   periodicity = false)
    function refine_fn(p4est, which_tree, quadrant)
        quadrant_obj = unsafe_load(quadrant)
        if quadrant_obj.x == 0 && quadrant_obj.y == 0 && quadrant_obj.level < 4
            return Cint(1) # refine
        else
            return Cint(0) # do not refine
        end
    end
    refine_fn_c = @cfunction(refine_fn, Cint,
                             (Ptr{Trixi.p4est_t}, Ptr{Trixi.p4est_topidx_t},
                              Ptr{Trixi.p4est_quadrant_t}))
    Trixi.refine_p4est!(mesh_nonconforming.p4est, true, refine_fn_c, C_NULL)

    @test_trixi_include(joinpath(EXAMPLES_DIR,
                                 "elixir_mhd_alfven_wave_combined_fluxes_nonperiodic.jl"),
                        # Reference errors computed on the CPU with `Float64` for this
                        # mesh, which is not used by any CPU test item
                        l2=Float32[5.581296218772088e-5,
                                   5.16360786373411e-5,
                                   5.12882108402347e-5,
                                   0.0001229940026020629,
                                   6.985260277446713e-5,
                                   7.235772483477128e-5,
                                   7.278163788909123e-5,
                                   0.00012154873069419296,
                                   5.331127637791119e-5],
                        linf=Float32[0.00032478943961145035,
                                     0.00043031387863581155,
                                     0.00043576215760957743,
                                     0.0010599638007024459,
                                     0.00043320839902183383,
                                     0.0005048047383285903,
                                     0.0005115349840848493,
                                     0.0010574376423485438,
                                     0.00023609057563457795],
                        RealT_for_test_tolerances=Float32,
                        mesh=mesh_nonconforming,
                        tspan=(0.0, 0.25),
                        real_type=Float32,
                        storage_type=MtlArray)
    # The elixir adapts `semi` before creating the ODE problem
    @test semi === ode.p
    @test real(semi.solver) == Float32
    @test real(semi.solver.basis) == Float32
    @test real(semi.solver.mortar) == Float32
    # TODO: `mesh` is currently not `adapt`ed correctly
    @test real(semi.mesh) == Float64
    @test typeof(semi.equations.gamma) == Float32

    # Make sure that the mortar kernels are used with the combined conservative
    # and nonconservative fluxes
    @test Trixi.nmortars(semi.solver, semi.cache) > 0
    @test Trixi.combine_conservative_and_nonconservative_fluxes(semi.solver.surface_integral.surface_flux,
                                                                semi.equations) ==
          Trixi.True()
    @test ode.u0 isa MtlArray
    @test semi.solver.basis.derivative_matrix isa MtlArray

    @test Trixi.storage_type(semi.cache.elements) === MtlArray
    @test Trixi.storage_type(semi.cache.interfaces) === MtlArray
    @test Trixi.storage_type(semi.cache.boundaries) === MtlArray
    @test Trixi.storage_type(semi.cache.mortars) === MtlArray

    # Ensure that the RHS computation overwrites existing data in `du` correctly.
    u_ode = copy(ode.u0)
    du_ode = similar(u_ode)
    fill!(du_ode, convert(eltype(du_ode), NaN))
    Trixi.rhs_hyperbolic!(du_ode, u_ode, semi, first(ode.tspan))
    @test all(isfinite, du_ode)
end

@testitem "Metal 2D: elixir_navierstokes_lid_driven_cavity.jl Float32 / Metal" setup=[
    Setup,
    Metal2DExamples
] tags=[:Metal] begin
    # Using Metal inside the testitem since otherwise the bindings are hidden by the anonymous modules
    using Metal
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_navierstokes_lid_driven_cavity.jl"),
                        initial_refinement_level=2, tspan=(0.0, 0.5),
                        # The default `time_int_tol = 1e-8` of the adaptive time
                        # integrator is below the machine epsilon of `Float32`.
                        # Then, the step size controller reacts to round-off
                        # errors and the results depend on the hardware.
                        # Metal does not support `Float64`, so the tolerance must
                        # be a `Float32` to be usable in the GPU kernels.
                        time_int_tol=1.0f-6,
                        l2=Float32[0.00028740498,
                                   0.0810115,
                                   0.020995662,
                                   0.04966808],
                        linf=Float32[0.014813244,
                                     0.95132273,
                                     0.722385,
                                     1.4834595],
                        RealT_for_test_tolerances=Float32,
                        real_type=Float32,
                        storage_type=MtlArray)
    # The elixir adapts `semi` before creating the ODE problem
    @test semi === ode.p
    @test real(semi.solver) == Float32
    @test typeof(semi.equations_parabolic.mu) == Float32
    @test ode.u0 isa MtlArray{Float32}
    @test semi.cache_parabolic.parabolic_container.u_transformed isa MtlArray{Float32}

    # Ensure that the RHS computation overwrites existing data in `du` correctly.
    u_ode = copy(ode.u0)
    du_ode = similar(u_ode)
    fill!(du_ode, convert(eltype(du_ode), NaN))
    Trixi.rhs_hyperbolic!(du_ode, u_ode, semi, first(ode.tspan))
    @test all(isfinite, du_ode)
    fill!(du_ode, convert(eltype(du_ode), NaN))
    Trixi.rhs_parabolic!(du_ode, u_ode, semi, first(ode.tspan))
    @test all(isfinite, du_ode)
end
