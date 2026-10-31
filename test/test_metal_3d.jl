# Test items tagged `:metal_skip_m1_m2` fail on Apple M1/M2 GPUs because of upstream
# bugs in Metal.jl and are skipped there, see `runtests.jl`.

@testsnippet Metal3DExamples begin
    EXAMPLES_DIR = joinpath(examples_dir(), "p4est_3d_dgsem")
end

@testitem "Metal 3D: elixir_advection_basic.jl Float32 / Metal" setup=[
    Setup,
    Metal3DExamples
] tags=[:Metal, :metal_skip_m1_m2] begin
    # Using Metal inside the testitem since otherwise the bindings are hidden by the anonymous modules
    using Metal
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_basic.jl"),
                        # Expected errors similar to reference on CPU
                        l2=[Float32(0.00016263963870641478)],
                        linf=[Float32(0.0014537194925779984)],
                        RealT_for_test_tolerances=Float32,
                        real_type=Float32,
                        storage_type=MtlArray)
    # The elixir adapts `semi` before creating the ODE problem
    @test semi === ode.p
    @test real(semi.solver) == Float32
    @test real(semi.solver.basis) == Float32
    @test real(semi.solver.mortar) == Float32
    # TODO: remake ignores the mesh itself as well
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

@testitem "Metal 3D: elixir_euler_source_terms_nonperiodic.jl Float32 / Metal" setup=[
    Setup,
    Metal3DExamples
] tags=[:Metal, :metal_skip_m1_m2] begin
    # Using Metal inside the testitem since otherwise the bindings are hidden by the anonymous modules
    using Metal
    @test_trixi_include(joinpath(EXAMPLES_DIR,
                                 "elixir_euler_source_terms_nonperiodic.jl"),
                        l2=Float32[0.0014518665391031068,
                                   0.0014470701356811022,
                                   0.0014470866449955344,
                                   0.00144707575575548,
                                   0.0029342928549885568],
                        linf=Float32[0.010317440030529479,
                                     0.011303550618318114,
                                     0.011295533976851013,
                                     0.011299068214785102,
                                     0.0209091211162149],
                        volume_integral=VolumeIntegralFluxDifferencing(flux_kennedy_gruber),
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

@testitem "Metal 3D: elixir_mhd_alfven_wave_combined_fluxes_nonperiodic.jl Float32 / Metal" setup=[
    Setup,
    Metal3DExamples
] tags=[:Metal, :metal_skip_m1_m2] begin
    # Using Metal inside the testset since otherwise the bindings are hiddend by the anonymous modules
    using Metal
    using Trixi
    @test_trixi_include(joinpath(EXAMPLES_DIR,
                                 "elixir_mhd_alfven_wave_combined_fluxes_nonperiodic.jl"),
                        l2=Float32[0.00020525117656966933,
                                   0.0006360228815289537,
                                   0.0002768843174099056,
                                   0.0007803905033283389,
                                   0.0007096877693750141,
                                   0.0006334907401971527,
                                   0.00031746218801771323,
                                   0.0007723227422473114,
                                   0.00023680544707528988],
                        linf=Float32[0.0011634305119514465,
                                     0.004142922370076324,
                                     0.0016762695564673177,
                                     0.004706167449922888,
                                     0.004352995157241857,
                                     0.0038981913494133824,
                                     0.0014574094164283569,
                                     0.004453214425426177,
                                     0.0011997520958011896],
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

@testitem "Metal 3D: elixir_advection_nonconforming.jl Float32 / Metal" setup=[
    Setup,
    Metal3DExamples
] tags=[:Metal, :metal_skip_m1_m2] begin
    # Using Metal inside the testitem since otherwise the bindings are hidden by the anonymous modules
    using Metal
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_nonconforming.jl"),
                        # Expected errors similar to reference on CPU
                        l2=Float32[0.00253595715323843],
                        linf=Float32[0.016486952252155795],
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

@testitem "Metal 3D: elixir_euler_source_terms_nonconforming_unstructured_curved.jl Float32 / Metal" setup=[
    Setup,
    Metal3DExamples
] tags=[:Metal, :metal_skip_m1_m2] begin
    # Using Metal inside the testitem since otherwise the bindings are hidden by the anonymous modules
    using Metal
    @test_trixi_include(joinpath(EXAMPLES_DIR,
                                 "elixir_euler_source_terms_nonconforming_unstructured_curved.jl"),
                        # Expected errors similar to reference on CPU
                        l2=Float32[4.070355207909268e-5,
                                   4.4993257426833716e-5,
                                   5.10588457841744e-5,
                                   5.102840924036687e-5,
                                   0.00019986264001630542],
                        linf=Float32[0.0016987332417202072,
                                     0.003622956808262634,
                                     0.002029576258317789,
                                     0.0024206977281964193,
                                     0.008526972236273522],
                        tspan=(0.0, 0.01),
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
    @test any(indices -> :i_backward in indices || :j_backward in indices,
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

@testitem "Metal 3D: elixir_mhd_alfven_wave_combined_fluxes_nonperiodic.jl nonconforming Float32 / Metal" setup=[
    Setup,
    Metal3DExamples
] tags=[:Metal, :metal_skip_m1_m2] begin
    # Using Metal inside the testitem since otherwise the bindings are hidden by the anonymous modules
    using Metal
    using Trixi
    # Create a non-conforming mesh by refining the bottom left front octant of each tree
    # to level 4 as in `elixir_mhd_alfven_wave_nonconforming.jl`, but without
    # periodicity to match the boundary conditions of the elixir used below
    mesh_nonconforming = P4estMesh((2, 2, 2), polydeg = 3, initial_refinement_level = 1,
                                   coordinates_min = (-1.0, -1.0, -1.0),
                                   coordinates_max = (1.0, 1.0, 1.0),
                                   periodicity = false)
    function refine_fn(p8est, which_tree, quadrant)
        quadrant_obj = unsafe_load(quadrant)
        if quadrant_obj.x == 0 && quadrant_obj.y == 0 && quadrant_obj.z == 0 &&
           quadrant_obj.level < 3
            # return true (refine)
            return Cint(1)
        else
            # return false (don't refine)
            return Cint(0)
        end
    end
    refine_fn_c = @cfunction(refine_fn, Cint,
                             (Ptr{Trixi.p8est_t}, Ptr{Trixi.p4est_topidx_t},
                              Ptr{Trixi.p8est_quadrant_t}))
    Trixi.refine_p4est!(mesh_nonconforming.p4est, true, refine_fn_c, C_NULL)

    @test_trixi_include(joinpath(EXAMPLES_DIR,
                                 "elixir_mhd_alfven_wave_combined_fluxes_nonperiodic.jl"),
                        # Reference errors computed with KernelAbstractions CPU backend with `Float64`
                        l2=Float32[0.001656350469788179,
                                   0.005878912164586573,
                                   0.0030351693310068915,
                                   0.006415910708712904,
                                   0.006315360797389399,
                                   0.006235810541664033,
                                   0.002844337960841682,
                                   0.006547611570440981,
                                   0.0033770756724939297],
                        linf=Float32[0.011726707425792626,
                                     0.043758334597288714,
                                     0.018828037099486855,
                                     0.044739187284319026,
                                     0.047655270947795536,
                                     0.04068120133668496,
                                     0.02570557246571159,
                                     0.04804236486857274,
                                     0.024268494473398607],
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

@testitem "Metal 3D: elixir_euler_source_terms_nonperiodic.jl HalfSweep vs. FullSweep vs. FullSweepGlobal / Metal" setup=[
    Setup,
    Metal3DExamples
] tags=[:Metal, :metal_skip_m1_m2] begin
    # Using Metal inside the testitem since otherwise the bindings are hidden by the anonymous modules
    using Metal

    trixi_include(@__MODULE__,
                  joinpath(EXAMPLES_DIR, "elixir_euler_source_terms_nonperiodic.jl"),
                  volume_integral = VolumeIntegralFluxDifferencing(flux_kennedy_gruber),
                  tspan = (0.0, 0.5),
                  real_type = Float32,
                  storage_type = MtlArray,
                  tuning = Tuning(flux_differencing_kernel = HalfSweep()))
    @test semi === ode.p
    @test semi.cache.tuning.flux_differencing_kernel === HalfSweep()
    @test Trixi.storage_type(semi.cache.elements) === MtlArray
    u_half_sweep = Array(sol.u[end])

    # Ensure that the RHS computation overwrites existing data in `du` correctly.
    u_ode = copy(ode.u0)
    du_ode = similar(u_ode)
    fill!(du_ode, convert(eltype(du_ode), NaN))
    Trixi.rhs_hyperbolic!(du_ode, u_ode, semi, first(ode.tspan))
    @test all(isfinite, du_ode)

    trixi_include(@__MODULE__,
                  joinpath(EXAMPLES_DIR, "elixir_euler_source_terms_nonperiodic.jl"),
                  volume_integral = VolumeIntegralFluxDifferencing(flux_kennedy_gruber),
                  tspan = (0.0, 0.5),
                  real_type = Float32,
                  storage_type = MtlArray,
                  tuning = Tuning(flux_differencing_kernel = FullSweep()))
    @test semi === ode.p
    @test semi.cache.tuning.flux_differencing_kernel === FullSweep()
    u_full_sweep = Array(sol.u[end])

    # Ensure that the RHS computation overwrites existing data in `du` correctly.
    u_ode = copy(ode.u0)
    du_ode = similar(u_ode)
    fill!(du_ode, convert(eltype(du_ode), NaN))
    Trixi.rhs_hyperbolic!(du_ode, u_ode, semi, first(ode.tspan))
    @test all(isfinite, du_ode)

    trixi_include(@__MODULE__,
                  joinpath(EXAMPLES_DIR, "elixir_euler_source_terms_nonperiodic.jl"),
                  volume_integral = VolumeIntegralFluxDifferencing(flux_kennedy_gruber),
                  tspan = (0.0, 0.5),
                  real_type = Float32,
                  storage_type = MtlArray,
                  tuning = Tuning(flux_differencing_kernel = FullSweepGlobal()))
    @test semi === ode.p
    @test semi.cache.tuning.flux_differencing_kernel === FullSweepGlobal()
    u_full_sweep_global = Array(sol.u[end])

    # Ensure that the RHS computation overwrites existing data in `du` correctly.
    u_ode = copy(ode.u0)
    du_ode = similar(u_ode)
    fill!(du_ode, convert(eltype(du_ode), NaN))
    Trixi.rhs_hyperbolic!(du_ode, u_ode, semi, first(ode.tspan))
    @test all(isfinite, du_ode)

    @test u_half_sweep ≈ u_full_sweep
    @test u_half_sweep ≈ u_full_sweep_global
end

@testitem "Metal 3D: elixir_mhd_alfven_wave_combined_fluxes_nonperiodic.jl HalfSweep vs. FullSweep vs. FullSweepGlobal / Metal" setup=[
    Setup,
    Metal3DExamples
] tags=[:Metal, :metal_skip_m1_m2] begin
    # Using Metal inside the testitem since otherwise the bindings are hidden by the anonymous modules
    using Metal

    trixi_include(@__MODULE__,
                  joinpath(EXAMPLES_DIR,
                           "elixir_mhd_alfven_wave_combined_fluxes_nonperiodic.jl"),
                  tspan = (0.0, 0.1),
                  real_type = Float32,
                  storage_type = MtlArray,
                  tuning = Tuning(flux_differencing_kernel = HalfSweep()))
    @test semi === ode.p
    @test semi.cache.tuning.flux_differencing_kernel === HalfSweep()
    @test Trixi.storage_type(semi.cache.elements) === MtlArray
    u_half_sweep = Array(sol.u[end])

    # Ensure that the RHS computation overwrites existing data in `du` correctly.
    u_ode = copy(ode.u0)
    du_ode = similar(u_ode)
    fill!(du_ode, convert(eltype(du_ode), NaN))
    Trixi.rhs_hyperbolic!(du_ode, u_ode, semi, first(ode.tspan))
    @test all(isfinite, du_ode)

    trixi_include(@__MODULE__,
                  joinpath(EXAMPLES_DIR,
                           "elixir_mhd_alfven_wave_combined_fluxes_nonperiodic.jl"),
                  tspan = (0.0, 0.1),
                  real_type = Float32,
                  storage_type = MtlArray,
                  tuning = Tuning(flux_differencing_kernel = FullSweep()))
    @test semi === ode.p
    @test semi.cache.tuning.flux_differencing_kernel === FullSweep()
    u_full_sweep = Array(sol.u[end])

    # Ensure that the RHS computation overwrites existing data in `du` correctly.
    u_ode = copy(ode.u0)
    du_ode = similar(u_ode)
    fill!(du_ode, convert(eltype(du_ode), NaN))
    Trixi.rhs_hyperbolic!(du_ode, u_ode, semi, first(ode.tspan))
    @test all(isfinite, du_ode)

    trixi_include(@__MODULE__,
                  joinpath(EXAMPLES_DIR,
                           "elixir_mhd_alfven_wave_combined_fluxes_nonperiodic.jl"),
                  tspan = (0.0, 0.1),
                  real_type = Float32,
                  storage_type = MtlArray,
                  tuning = Tuning(flux_differencing_kernel = FullSweepGlobal()))
    @test semi === ode.p
    @test semi.cache.tuning.flux_differencing_kernel === FullSweepGlobal()
    u_full_sweep_global = Array(sol.u[end])

    # Ensure that the RHS computation overwrites existing data in `du` correctly.
    u_ode = copy(ode.u0)
    du_ode = similar(u_ode)
    fill!(du_ode, convert(eltype(du_ode), NaN))
    Trixi.rhs_hyperbolic!(du_ode, u_ode, semi, first(ode.tspan))
    @test all(isfinite, du_ode)

    @test u_half_sweep ≈ u_full_sweep
    @test u_half_sweep ≈ u_full_sweep_global
end

@testitem "Metal 3D: elixir_euler_source_terms_nonperiodic.jl FluxTurbo / Metal" setup=[
    Setup,
    Metal3DExamples
] tags=[:Metal, :metal_skip_m1_m2] begin
    # Using Metal inside the testitem since otherwise the bindings are hidden by the anonymous modules
    using Metal

    trixi_include(@__MODULE__,
                  joinpath(EXAMPLES_DIR, "elixir_euler_source_terms_nonperiodic.jl"),
                  volume_integral = VolumeIntegralFluxDifferencing(flux_ranocha),
                  tspan = (0.0, 0.5),
                  real_type = Float32,
                  storage_type = MtlArray,
                  tuning = Tuning(flux_differencing_kernel = HalfSweep()))
    @test semi === ode.p
    @test Trixi.storage_type(semi.cache.elements) === MtlArray
    u_plain = Array(sol.u[end])

    # Ensure that the RHS computation overwrites existing data in `du` correctly.
    u_ode = copy(ode.u0)
    du_ode = similar(u_ode)
    fill!(du_ode, convert(eltype(du_ode), NaN))
    Trixi.rhs_hyperbolic!(du_ode, u_ode, semi, first(ode.tspan))
    @test all(isfinite, du_ode)

    trixi_include(@__MODULE__,
                  joinpath(EXAMPLES_DIR, "elixir_euler_source_terms_nonperiodic.jl"),
                  volume_integral = VolumeIntegralFluxDifferencing(FluxTurbo(flux_ranocha)),
                  tspan = (0.0, 0.5),
                  real_type = Float32,
                  storage_type = MtlArray,
                  tuning = Tuning(flux_differencing_kernel = HalfSweep()))
    @test semi === ode.p
    @test semi.cache.tuning.flux_differencing_kernel === HalfSweep()
    @test Array(sol.u[end]) ≈ u_plain

    # Ensure that the RHS computation overwrites existing data in `du` correctly.
    u_ode = copy(ode.u0)
    du_ode = similar(u_ode)
    fill!(du_ode, convert(eltype(du_ode), NaN))
    Trixi.rhs_hyperbolic!(du_ode, u_ode, semi, first(ode.tspan))
    @test all(isfinite, du_ode)

    trixi_include(@__MODULE__,
                  joinpath(EXAMPLES_DIR, "elixir_euler_source_terms_nonperiodic.jl"),
                  volume_integral = VolumeIntegralFluxDifferencing(FluxTurbo(flux_ranocha)),
                  tspan = (0.0, 0.5),
                  real_type = Float32,
                  storage_type = MtlArray,
                  tuning = Tuning(flux_differencing_kernel = FullSweep()))
    @test semi === ode.p
    @test semi.cache.tuning.flux_differencing_kernel === FullSweep()
    @test Array(sol.u[end]) ≈ u_plain

    # Ensure that the RHS computation overwrites existing data in `du` correctly.
    u_ode = copy(ode.u0)
    du_ode = similar(u_ode)
    fill!(du_ode, convert(eltype(du_ode), NaN))
    Trixi.rhs_hyperbolic!(du_ode, u_ode, semi, first(ode.tspan))
    @test all(isfinite, du_ode)

    trixi_include(@__MODULE__,
                  joinpath(EXAMPLES_DIR, "elixir_euler_source_terms_nonperiodic.jl"),
                  volume_integral = VolumeIntegralFluxDifferencing(FluxTurbo(flux_ranocha)),
                  tspan = (0.0, 0.5),
                  real_type = Float32,
                  storage_type = MtlArray,
                  tuning = Tuning(flux_differencing_kernel = FullSweepGlobal()))
    @test semi === ode.p
    @test semi.cache.tuning.flux_differencing_kernel === FullSweepGlobal()
    @test Array(sol.u[end]) ≈ u_plain

    # Ensure that the RHS computation overwrites existing data in `du` correctly.
    u_ode = copy(ode.u0)
    du_ode = similar(u_ode)
    fill!(du_ode, convert(eltype(du_ode), NaN))
    Trixi.rhs_hyperbolic!(du_ode, u_ode, semi, first(ode.tspan))
    @test all(isfinite, du_ode)
end

@testitem "Metal 3D: elixir_mhd_alfven_wave_combined_fluxes_nonperiodic.jl FluxTurbo / Metal" setup=[
    Setup,
    Metal3DExamples
] tags=[:Metal, :metal_skip_m1_m2] begin
    # Using Metal inside the testitem since otherwise the bindings are hidden by the anonymous modules
    using Metal

    volume_flux_turbo = FluxTurbo(flux_hindenlang_gassner, flux_nonconservative_powell)

    trixi_include(@__MODULE__,
                  joinpath(EXAMPLES_DIR,
                           "elixir_mhd_alfven_wave_combined_fluxes_nonperiodic.jl"),
                  tspan = (0.0, 0.1),
                  real_type = Float32,
                  storage_type = MtlArray,
                  tuning = Tuning(flux_differencing_kernel = HalfSweep()))
    @test semi === ode.p
    u_plain = Array(sol.u[end])

    # Ensure that the RHS computation overwrites existing data in `du` correctly.
    u_ode = copy(ode.u0)
    du_ode = similar(u_ode)
    fill!(du_ode, convert(eltype(du_ode), NaN))
    Trixi.rhs_hyperbolic!(du_ode, u_ode, semi, first(ode.tspan))
    @test all(isfinite, du_ode)

    trixi_include(@__MODULE__,
                  joinpath(EXAMPLES_DIR,
                           "elixir_mhd_alfven_wave_combined_fluxes_nonperiodic.jl"),
                  volume_integral = VolumeIntegralFluxDifferencing(volume_flux_turbo),
                  tspan = (0.0, 0.1),
                  real_type = Float32,
                  storage_type = MtlArray,
                  tuning = Tuning(flux_differencing_kernel = HalfSweep()))
    @test semi === ode.p
    @test semi.cache.tuning.flux_differencing_kernel === HalfSweep()
    @test Array(sol.u[end]) ≈ u_plain

    # Ensure that the RHS computation overwrites existing data in `du` correctly.
    u_ode = copy(ode.u0)
    du_ode = similar(u_ode)
    fill!(du_ode, convert(eltype(du_ode), NaN))
    Trixi.rhs_hyperbolic!(du_ode, u_ode, semi, first(ode.tspan))
    @test all(isfinite, du_ode)

    trixi_include(@__MODULE__,
                  joinpath(EXAMPLES_DIR,
                           "elixir_mhd_alfven_wave_combined_fluxes_nonperiodic.jl"),
                  volume_integral = VolumeIntegralFluxDifferencing(volume_flux_turbo),
                  tspan = (0.0, 0.1),
                  real_type = Float32,
                  storage_type = MtlArray,
                  tuning = Tuning(flux_differencing_kernel = FullSweep()))
    @test semi === ode.p
    @test semi.cache.tuning.flux_differencing_kernel === FullSweep()
    @test Array(sol.u[end]) ≈ u_plain

    # Ensure that the RHS computation overwrites existing data in `du` correctly.
    u_ode = copy(ode.u0)
    du_ode = similar(u_ode)
    fill!(du_ode, convert(eltype(du_ode), NaN))
    Trixi.rhs_hyperbolic!(du_ode, u_ode, semi, first(ode.tspan))
    @test all(isfinite, du_ode)
end

@testitem "Metal 3D: elixir_navierstokes_convergence.jl Float32 / Metal" setup=[
    Setup,
    Metal3DExamples
] tags=[:Metal, :metal_skip_m1_m2] begin
    # Using Metal inside the testitem since otherwise the bindings are hidden by the anonymous modules
    using Metal
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_navierstokes_convergence.jl"),
                        initial_refinement_level=2, tspan=(0.0, 0.1),
                        # The default `time_int_tol = 1e-8` of the adaptive time
                        # integrator is below the machine epsilon of `Float32`.
                        # Then, the step size controller reacts to round-off
                        # errors and the results depend on the hardware.
                        # Metal does not support `Float64`, so the tolerance must
                        # be a `Float32` to be usable in the GPU kernels.
                        time_int_tol=1.0f-6,
                        l2=Float32[0.00026599237563895863,
                                   0.0004618770409481977,
                                   0.0005424902199947988,
                                   0.00046187821240216285,
                                   0.0015847432617011525],
                        linf=Float32[0.0025286846703638233,
                                     0.0063080801628530025,
                                     0.004338984007198565,
                                     0.006308289710432291,
                                     0.031785004991078836],
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
