@testsnippet CUDA2DExamples begin
    EXAMPLES_DIR = joinpath(examples_dir(), "p4est_2d_dgsem")
end

@testitem "CUDA 2D: elixir_advection_basic.jl native" setup=[Setup, CUDA2DExamples] tags=[:CUDA] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_basic.jl"),
                        # Expected errors are exactly the same as with TreeMesh!
                        l2=8.311947673061856e-6,
                        linf=6.627000273229378e-5)
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    semi = ode.p # `semidiscretize` adapts the semi, so we need to obtain it from the ODE problem.
    @test real(ode.p.solver) == Float64
    @test real(ode.p.solver.basis) == Float64
    @test real(ode.p.solver.mortar) == Float64
    # TODO: `mesh` is currently not `adapt`ed correctly
    @test real(ode.p.mesh) == Float64
    @test eltype(ode.p.equations.advection_velocity) == Float64

    @test ode.u0 isa Array
    @test ode.p.solver.basis.derivative_matrix isa Array

    @test Trixi.storage_type(ode.p.cache.elements) === Array
    @test Trixi.storage_type(ode.p.cache.interfaces) === Array
    @test Trixi.storage_type(ode.p.cache.boundaries) === Array
    @test Trixi.storage_type(ode.p.cache.mortars) === Array

    # Ensure that the RHS computation overwrites existing data in `du` correctly.
    u_ode = copy(ode.u0)
    du_ode = similar(u_ode)
    fill!(du_ode, convert(eltype(du_ode), NaN))
    Trixi.rhs_hyperbolic!(du_ode, u_ode, ode.p, first(ode.tspan))
    @test all(isfinite, du_ode)
end

@testitem "CUDA 2D: elixir_advection_basic.jl Float32 / CUDA" setup=[Setup, CUDA2DExamples] tags=[:CUDA] begin
    # Using CUDA inside the testitem since otherwise the bindings are hidden by the anonymous modules
    using CUDA
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_basic.jl"),
                        # Expected errors are exactly the same as with TreeMesh!
                        l2=[Float32(8.311947673061856e-6)],
                        linf=[Float32(6.627000273229378e-5)],
                        RealT_for_test_tolerances=Float32,
                        real_type=Float32,
                        storage_type=CuArray)
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    semi = ode.p # `semidiscretize` adapts the semi, so we need to obtain it from the ODE problem.
    @test real(ode.p.solver) == Float32
    @test real(ode.p.solver.basis) == Float32
    @test real(ode.p.solver.mortar) == Float32
    # TODO: `mesh` is currently not `adapt`ed correctly
    @test real(ode.p.mesh) == Float64
    @test eltype(ode.p.equations.advection_velocity) == Float32

    @test ode.u0 isa CuArray
    @test ode.p.solver.basis.derivative_matrix isa CuArray

    @test Trixi.storage_type(ode.p.cache.elements) === CuArray
    @test Trixi.storage_type(ode.p.cache.interfaces) === CuArray
    @test Trixi.storage_type(ode.p.cache.boundaries) === CuArray
    @test Trixi.storage_type(ode.p.cache.mortars) === CuArray

    # Ensure that the RHS computation overwrites existing data in `du` correctly.
    u_ode = copy(ode.u0)
    du_ode = similar(u_ode)
    fill!(du_ode, convert(eltype(du_ode), NaN))
    Trixi.rhs_hyperbolic!(du_ode, u_ode, ode.p, first(ode.tspan))
    @test all(isfinite, du_ode)
end

@testitem "CUDA 2D: elixir_euler_source_terms.jl native" setup=[Setup, CUDA2DExamples] tags=[:CUDA] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_source_terms.jl"),
                        # Expected errors are exactly the same as with TreeMesh!
                        l2=[9.321181254378498e-7,
                            1.418121074369651e-6,
                            1.4181210743821669e-6,
                            4.824553091168877e-6],
                        linf=[9.577246532499473e-6,
                            1.1707525985116263e-5,
                            1.1707525982673772e-5,
                            4.886961559069647e-5])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    semi = ode.p # `semidiscretize` adapts the semi, so we need to obtain it from the ODE problem.
    @test real(semi.solver) == Float64
    @test real(semi.solver.basis) == Float64
    @test real(semi.solver.mortar) == Float64
    # TODO: `mesh` is currently not `adapt`ed correctly
    @test real(semi.mesh) == Float64
    @test typeof(semi.equations.gamma) == Float64

    @test ode.u0 isa Array
    @test semi.solver.basis.derivative_matrix isa Array

    @test Trixi.storage_type(semi.cache.elements) === Array
    @test Trixi.storage_type(semi.cache.interfaces) === Array
    @test Trixi.storage_type(semi.cache.boundaries) === Array
    @test Trixi.storage_type(semi.cache.mortars) === Array

    # Ensure that the RHS computation overwrites existing data in `du` correctly.
    u_ode = copy(ode.u0)
    du_ode = similar(u_ode)
    fill!(du_ode, convert(eltype(du_ode), NaN))
    Trixi.rhs_hyperbolic!(du_ode, u_ode, ode.p, first(ode.tspan))
    @test all(isfinite, du_ode)
end

@testitem "CUDA 2D: elixir_euler_source_terms.jl Float32 / CUDA" setup=[
    Setup,
    CUDA2DExamples
] tags=[:CUDA] begin
    # Using CUDA inside the testitem since otherwise the bindings are hidden by the anonymous modules
    using CUDA
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
                        storage_type=CuArray)
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    semi = ode.p # `semidiscretize` adapts the semi, so we need to obtain it from the ODE problem.
    @test real(semi.solver) == Float32
    @test real(semi.solver.basis) == Float32
    @test real(semi.solver.mortar) == Float32
    # TODO: `mesh` is currently not `adapt`ed correctly
    @test real(semi.mesh) == Float64
    @test typeof(semi.equations.gamma) == Float32

    @test ode.u0 isa CuArray
    @test semi.solver.basis.derivative_matrix isa CuArray

    @test Trixi.storage_type(semi.cache.elements) === CuArray
    @test Trixi.storage_type(semi.cache.interfaces) === CuArray
    @test Trixi.storage_type(semi.cache.boundaries) === CuArray
    @test Trixi.storage_type(semi.cache.mortars) === CuArray

    # Ensure that the RHS computation overwrites existing data in `du` correctly.
    u_ode = copy(ode.u0)
    du_ode = similar(u_ode)
    fill!(du_ode, convert(eltype(du_ode), NaN))
    Trixi.rhs_hyperbolic!(du_ode, u_ode, ode.p, first(ode.tspan))
    @test all(isfinite, du_ode)
end

@testitem "CUDA 2D: elixir_euler_source_terms.jl Flux Differencing Float32 / CUDA" setup=[
    Setup,
    CUDA2DExamples
] tags=[:CUDA] begin
    # Using CUDA inside the testitem since otherwise the bindings are hidden by the anonymous modules
    using CUDA
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
                        storage_type=CuArray,
                        solver=DGSEM(polydeg = 3,
                                     surface_flux = FluxLaxFriedrichs(max_abs_speed_naive),
                                     volume_integral = VolumeIntegralFluxDifferencing(flux_kennedy_gruber)))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    semi = ode.p # `semidiscretize` adapts the semi, so we need to obtain it from the ODE problem.
    @test real(semi.solver) == Float32
    @test real(semi.solver.basis) == Float32
    @test real(semi.solver.mortar) == Float32
    # TODO: `mesh` is currently not `adapt`ed correctly
    @test real(semi.mesh) == Float64
    @test typeof(semi.equations.gamma) == Float32

    @test ode.u0 isa CuArray
    @test semi.solver.basis.derivative_matrix isa CuArray

    @test Trixi.storage_type(semi.cache.elements) === CuArray
    @test Trixi.storage_type(semi.cache.interfaces) === CuArray
    @test Trixi.storage_type(semi.cache.boundaries) === CuArray
    @test Trixi.storage_type(semi.cache.mortars) === CuArray

    # Ensure that the RHS computation overwrites existing data in `du` correctly.
    u_ode = copy(ode.u0)
    du_ode = similar(u_ode)
    fill!(du_ode, convert(eltype(du_ode), NaN))
    Trixi.rhs_hyperbolic!(du_ode, u_ode, ode.p, first(ode.tspan))
    @test all(isfinite, du_ode)
end

@testitem "CUDA 2D: elixir_mhd_alfven_wave_combined_fluxes_nonperiodic.jl Float32 / CUDA" setup=[
    Setup,
    CUDA2DExamples
] tags=[:CUDA] begin
    # Using CUDA inside the testitem since otherwise the bindings are hidden by the anonymous modules
    using CUDA
    using Trixi
    @test_trixi_include(joinpath(EXAMPLES_DIR,
                                 "elixir_mhd_alfven_wave_combined_fluxes_nonperiodic.jl"),
                        l2=Float32[8.281976064899433e-5,
                                   6.674408302881695e-5,
                                   6.693536534139316e-5,
                                   0.00011717744999013579,
                                   6.889569500245608e-5,
                                   7.78292854879118e-5,
                                   7.820255919638926e-5,
                                   0.00011506970727212514,
                                   5.3791801822110654e-5],
                        linf=Float32[0.00043082237243652344,
                                     0.0005365351910699076,
                                     0.0005327751111221801,
                                     0.0009163264949127586,
                                     0.00042850648667691615,
                                     0.0005048022425613308,
                                     0.0005058775894211109,
                                     0.0008949209768577965,
                                     0.00018917795326144592],
                        RealT_for_test_tolerances=Float32,
                        real_type=Float32,
                        storage_type=CuArray)
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    semi = ode.p # `semidiscretize` adapts the semi, so we need to obtain it from the ODE problem.
    @test real(semi.solver) == Float32
    @test real(semi.solver.basis) == Float32
    @test real(semi.solver.mortar) == Float32
    # TODO: `mesh` is currently not `adapt`ed correctly
    @test real(semi.mesh) == Float64
    @test typeof(semi.equations.gamma) == Float32

    @test ode.u0 isa CuArray
    @test semi.solver.basis.derivative_matrix isa CuArray

    @test Trixi.storage_type(semi.cache.elements) === CuArray
    @test Trixi.storage_type(semi.cache.interfaces) === CuArray
    @test Trixi.storage_type(semi.cache.boundaries) === CuArray
    @test Trixi.storage_type(semi.cache.mortars) === CuArray

    # Ensure that the RHS computation overwrites existing data in `du` correctly.
    u_ode = copy(ode.u0)
    du_ode = similar(u_ode)
    fill!(du_ode, convert(eltype(du_ode), NaN))
    Trixi.rhs_hyperbolic!(du_ode, u_ode, ode.p, first(ode.tspan))
    @test all(isfinite, du_ode)
end

@testitem "CUDA 2D: elixir_navierstokes_convergence.jl Float64 / CUDA" setup=[
    Setup,
    CUDA2DExamples
] tags=[:CUDA] begin
    # Using CUDA inside the testitem since otherwise the bindings are hidden by the anonymous modules
    using CUDA
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_navierstokes_convergence.jl"),
                        initial_refinement_level=1, tspan=(0.0, 0.2),
                        # Expected errors are exactly the same as in the parabolic test!
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
                        ],
                        storage_type=CuArray)
    semi = ode.p # `semidiscretize` adapts the semi, so we need to obtain it from the ODE problem.
    @test real(semi.solver) == Float64
    @test typeof(semi.equations_parabolic.mu) == Float64

    @test ode.u0 isa CuArray
    @test semi.solver.basis.derivative_matrix isa CuArray

    @test Trixi.storage_type(semi.cache.elements) === CuArray
    @test Trixi.storage_type(semi.cache.interfaces) === CuArray
    @test Trixi.storage_type(semi.cache.boundaries) === CuArray
    @test Trixi.storage_type(semi.cache.mortars) === CuArray
    @test semi.cache_parabolic.parabolic_container.u_transformed isa CuArray
    @test all(x -> x isa CuArray, semi.cache_parabolic.parabolic_container.gradients)
    @test all(x -> x isa CuArray,
              semi.cache_parabolic.parabolic_container.flux_parabolic)

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

@testitem "CUDA 2D: elixir_navierstokes_lid_driven_cavity.jl Float64 / CUDA" setup=[
    Setup,
    CUDA2DExamples
] tags=[:CUDA] begin
    # Using CUDA inside the testitem since otherwise the bindings are hidden by the anonymous modules
    using CUDA
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_navierstokes_lid_driven_cavity.jl"),
                        initial_refinement_level=2, tspan=(0.0, 0.5),
                        # Expected errors are exactly the same as in the parabolic test!
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
                        ],
                        storage_type=CuArray)
    semi = ode.p # `semidiscretize` adapts the semi, so we need to obtain it from the ODE problem.
    @test ode.u0 isa CuArray
    @test semi.cache_parabolic.parabolic_container.u_transformed isa CuArray

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

@testitem "CUDA 2D: elixir_navierstokes_lid_driven_cavity.jl Float32 / CUDA" setup=[
    Setup,
    CUDA2DExamples
] tags=[:CUDA] begin
    # Using CUDA inside the testitem since otherwise the bindings are hidden by the anonymous modules
    using CUDA
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_navierstokes_lid_driven_cavity.jl"),
                        initial_refinement_level=2, tspan=(0.0, 0.5),
                        l2=Float32[0.00031915737375335507,
                                   0.08100071821161867,
                                   0.02099960476349156,
                                   0.04866932200947667],
                        linf=Float32[0.014976143836975098,
                                     0.9511703472393689,
                                     0.7222638225794071,
                                     1.4565359767599375],
                        RealT_for_test_tolerances=Float32,
                        real_type=Float32,
                        storage_type=CuArray)
    semi = ode.p # `semidiscretize` adapts the semi, so we need to obtain it from the ODE problem.
    @test real(semi.solver) == Float32
    @test typeof(semi.equations_parabolic.mu) == Float32
    @test ode.u0 isa CuArray{Float32}
    @test semi.cache_parabolic.parabolic_container.u_transformed isa CuArray{Float32}

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
