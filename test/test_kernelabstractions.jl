@testsnippet KernelAbstractionsExamples begin
    EXAMPLES_DIR = examples_dir()
end

@testitem "KernelAbstractions backend preference" setup=[Setup] tags=[:kernelabstractions] begin
    @test Trixi._PREFERENCE_THREADING == :kernelabstractions
end

@testitem "KernelAbstractions CPU 2D: elixir_advection_basic.jl" setup=[
    Setup,
    KernelAbstractionsExamples
] tags=[:kernelabstractions] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "p4est_2d_dgsem",
                                 "elixir_advection_basic.jl"),
                        # Expected errors are exactly the same as with TreeMesh!
                        l2=8.311947673061856e-6,
                        linf=6.627000273229378e-5)
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    # The elixir adapts `semi` before creating the ODE problem
    @test semi === ode.p
    @test real(semi.solver) == Float64
    @test real(semi.solver.basis) == Float64
    @test real(semi.solver.mortar) == Float64
    # TODO: `mesh` is currently not `adapt`ed correctly
    @test real(semi.mesh) == Float64
    @test typeof(semi.equations.advection_velocity) == SVector{2, Float64}

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

@testitem "KernelAbstractions CPU 2D: elixir_advection_basic.jl Float32" setup=[
    Setup,
    KernelAbstractionsExamples
] tags=[:kernelabstractions] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "p4est_2d_dgsem",
                                 "elixir_advection_basic.jl"),
                        # Expected errors similar to reference on CPU
                        l2=[Float32(8.311947673061856e-6)],
                        linf=[Float32(6.627000273229378e-5)],
                        RealT_for_test_tolerances=Float32,
                        real_type=Float32)
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    # The elixir adapts `semi` before creating the ODE problem
    @test semi === ode.p
    @test real(semi.solver) == Float32
    @test real(semi.solver.basis) == Float32
    @test real(semi.solver.mortar) == Float32
    # TODO: `mesh` is currently not `adapt`ed correctly
    @test real(semi.mesh) == Float64
    @test typeof(semi.equations.advection_velocity) == SVector{2, Float32}

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

@testitem "KernelAbstractions CPU 2D: elixir_euler_source_terms.jl native" setup=[
    Setup,
    KernelAbstractionsExamples
] tags=[:kernelabstractions] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "p4est_2d_dgsem",
                                 "elixir_euler_source_terms.jl"),
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
    # The elixir adapts `semi` before creating the ODE problem
    @test semi === ode.p
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

@testitem "KernelAbstractions CPU 2D: elixir_euler_source_terms.jl Float32" setup=[
    Setup,
    KernelAbstractionsExamples
] tags=[:kernelabstractions] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "p4est_2d_dgsem",
                                 "elixir_euler_source_terms.jl"),
                        l2=Float32[2.4917018095933837e-6,
                                   2.7148269885239423e-6,
                                   2.695290306860358e-6,
                                   6.243861976167833e-6],
                        linf=Float32[1.6489475493930428e-5,
                                     1.7499923706143505e-5,
                                     1.893043518075288e-5,
                                     6.214141845717336e-5],
                        RealT_for_test_tolerances=Float32,
                        real_type=Float32)
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    # The elixir adapts `semi` before creating the ODE problem
    @test semi === ode.p
    @test real(semi.solver) == Float32
    @test real(semi.solver.basis) == Float32
    @test real(semi.solver.mortar) == Float32
    # TODO: `mesh` is currently not `adapt`ed correctly
    @test real(semi.mesh) == Float64
    @test typeof(semi.equations.gamma) == Float32

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

@testitem "KernelAbstractions CPU 2D: elixir_euler_source_terms.jl Flux Differencing Float32" setup=[
    Setup,
    KernelAbstractionsExamples
] tags=[:kernelabstractions] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "p4est_2d_dgsem",
                                 "elixir_euler_source_terms.jl"),
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
                        solver=DGSEM(polydeg = 3,
                                     surface_flux = FluxLaxFriedrichs(max_abs_speed_naive),
                                     volume_integral = VolumeIntegralFluxDifferencing(flux_kennedy_gruber)))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    # The elixir adapts `semi` before creating the ODE problem
    @test semi === ode.p
    @test real(semi.solver) == Float32
    @test real(semi.solver.basis) == Float32
    @test real(semi.solver.mortar) == Float32
    # TODO: `mesh` is currently not `adapt`ed correctly
    @test real(semi.mesh) == Float64
    @test typeof(semi.equations.gamma) == Float32

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

@testitem "KernelAbstractions CPU 2D: elixir_mhd_alfven_wave_combined_fluxes_nonperiodic.jl native" setup=[
    Setup,
    KernelAbstractionsExamples
] tags=[:kernelabstractions] begin
    using Trixi
    @test_trixi_include(joinpath(EXAMPLES_DIR, "p4est_2d_dgsem",
                                 "elixir_mhd_alfven_wave_combined_fluxes_nonperiodic.jl"),
                        l2=[
                            7.526028387805706e-5,
                            6.32127626310509e-5,
                            6.325543234769327e-5,
                            0.00011732589843058861,
                            7.481219049195874e-5,
                            7.911320579162387e-5,
                            7.935110683856884e-5,
                            0.00011470903279325104,
                            5.034941237145169e-5
                        ],
                        linf=[
                            0.00043327748904631314,
                            0.000516740973694016,
                            0.0005212064092713631,
                            0.0009207747658964149,
                            0.0004046247762601318,
                            0.0005735802836948078,
                            0.000543026062852392,
                            0.000894462605877569,
                            0.000298700326760806
                        ])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    # The elixir adapts `semi` before creating the ODE problem
    @test semi === ode.p
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

@testitem "KernelAbstractions CPU 2D: elixir_mhd_alfven_wave_combined_fluxes_nonperiodic.jl Float32" setup=[
    Setup,
    KernelAbstractionsExamples
] tags=[:kernelabstractions] begin
    using Trixi
    @test_trixi_include(joinpath(EXAMPLES_DIR, "p4est_2d_dgsem",
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
                        real_type=Float32)
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    # The elixir adapts `semi` before creating the ODE problem
    @test semi === ode.p
    @test real(semi.solver) == Float32
    @test real(semi.solver.basis) == Float32
    @test real(semi.solver.mortar) == Float32
    # TODO: `mesh` is currently not `adapt`ed correctly
    @test real(semi.mesh) == Float64
    @test typeof(semi.equations.gamma) == Float32

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

@testitem "KernelAbstractions CPU 2D: elixir_advection_nonconforming_flag.jl Float32" setup=[
    Setup,
    KernelAbstractionsExamples
] tags=[:kernelabstractions] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "p4est_2d_dgsem",
                                 "elixir_advection_nonconforming_flag.jl"),
                        l2=Float32[3.198940059144588e-5],
                        linf=Float32[0.00030636069494005547],
                        RealT_for_test_tolerances=Float32,
                        real_type=Float32)
    # The elixir adapts `semi` before creating the ODE problem
    @test semi === ode.p
    @test real(semi.solver) == Float32
    @test real(semi.solver.basis) == Float32
    @test real(semi.solver.mortar) == Float32
    # TODO: `mesh` is currently not `adapt`ed correctly
    @test real(semi.mesh) == Float64
    @test eltype(semi.equations.advection_velocity) == Float32

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
    Trixi.rhs_hyperbolic!(du_ode, u_ode, semi, first(ode.tspan))
    @test all(isfinite, du_ode)
end

@testitem "KernelAbstractions CPU 2D: elixir_mhd_alfven_wave_nonconforming.jl (combined fluxes)" setup=[
    Setup,
    KernelAbstractionsExamples
] tags=[:kernelabstractions] begin
    # Load the fluxes combining conservative and nonconservative terms,
    # which are equivalent to the fluxes used in the elixir below.
    # We do not need to run this simulation, so we replace `sol = solve(...)`
    # by `sol = nothing`.
    trixi_include(@__MODULE__,
                  joinpath(EXAMPLES_DIR, "p4est_2d_dgsem",
                           "elixir_mhd_alfven_wave_combined_fluxes_nonperiodic.jl"),
                  sol = nothing)

    # Use the same reference values as for the separate fluxes
    @test_trixi_include(joinpath(EXAMPLES_DIR, "p4est_2d_dgsem",
                                 "elixir_mhd_alfven_wave_nonconforming.jl"),
                        l2=[
                            0.03218873485054769,
                            0.036104237528919156,
                            0.03565782684252082,
                            0.05290171901682014,
                            0.03363060793396599,
                            0.03881219835771841,
                            0.03870276636879549,
                            0.05291025814280089,
                            0.022494653302015218
                        ],
                        linf=[
                            0.17957235749398837,
                            0.1403110675224628,
                            0.13219291731458857,
                            0.21691023663331951,
                            0.11730522693720624,
                            0.1452058059708733,
                            0.144259629027772,
                            0.21753737360583258,
                            0.07672851650342119
                        ],
                        solver=DGSEM(polydeg = 3,
                                     surface_flux = flux_hlle_nonconservative_powell,
                                     volume_integral = VolumeIntegralFluxDifferencing(flux_hindenlang_gassner_nonconservative_powell)),
                        tspan=(0.0, 0.25))

    # Make sure that the combined fluxes are used at mortars
    @test Trixi.nmortars(semi.solver, semi.cache) > 0
    @test Trixi.trixi_backend(ode.u0) isa Trixi.KernelAbstractions.CPU

    # Ensure that the RHS computation overwrites existing data in `du` correctly.
    u_ode = copy(ode.u0)
    du_ode = similar(u_ode)
    fill!(du_ode, convert(eltype(du_ode), NaN))
    Trixi.rhs_hyperbolic!(du_ode, u_ode, semi, first(ode.tspan))
    @test all(isfinite, du_ode)
end

@testitem "KernelAbstractions CPU 2D: elixir_navierstokes_convergence.jl" setup=[
    Setup,
    KernelAbstractionsExamples
] tags=[:kernelabstractions] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "p4est_2d_dgsem",
                                 "elixir_navierstokes_convergence.jl"),
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
                        ])
    semi = ode.p # `semidiscretize` adapts the semi, so we need to obtain it from the ODE problem.
    @test real(semi.solver) == Float64
    @test ode.u0 isa Array
    @test Trixi.trixi_backend(ode.u0) isa Trixi.KernelAbstractions.CPU
    @test semi.cache_parabolic.parabolic_container.u_transformed isa Array

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

@testitem "KernelAbstractions CPU 2D: elixir_navierstokes_lid_driven_cavity.jl" setup=[
    Setup,
    KernelAbstractionsExamples
] tags=[:kernelabstractions] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "p4est_2d_dgsem",
                                 "elixir_navierstokes_lid_driven_cavity.jl"),
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
                        ])
    semi = ode.p # `semidiscretize` adapts the semi, so we need to obtain it from the ODE problem.
    @test ode.u0 isa Array
    @test semi.cache_parabolic.parabolic_container.u_transformed isa Array

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

@testitem "KernelAbstractions CPU 3D: elixir_advection_basic.jl" setup=[
    Setup,
    KernelAbstractionsExamples
] tags=[:kernelabstractions] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "p4est_3d_dgsem",
                                 "elixir_advection_basic.jl"),
                        # Expected errors are exactly the same as with TreeMesh!
                        l2=[0.00016263963870641478],
                        linf=[0.0014537194925779984])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    # The elixir adapts `semi` before creating the ODE problem
    @test semi === ode.p
    @test real(semi.solver) == Float64
    @test real(semi.solver.basis) == Float64
    @test real(semi.solver.mortar) == Float64
    # TODO: `mesh` is currently not `adapt`ed correctly
    @test real(semi.mesh) == Float64
    @test typeof(semi.equations.advection_velocity) == SVector{3, Float64}

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

@testitem "KernelAbstractions CPU 3D: elixir_advection_basic.jl Float32" setup=[
    Setup,
    KernelAbstractionsExamples
] tags=[:kernelabstractions] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "p4est_3d_dgsem",
                                 "elixir_advection_basic.jl"),
                        # Expected errors similar to reference on CPU
                        l2=[Float32(0.00016263963870641478)],
                        linf=[Float32(0.0014537194925779984)],
                        RealT_for_test_tolerances=Float32,
                        real_type=Float32)
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    # The elixir adapts `semi` before creating the ODE problem
    @test semi === ode.p
    @test real(semi.solver) == Float32
    @test real(semi.solver.basis) == Float32
    @test real(semi.solver.mortar) == Float32
    # TODO: `mesh` is currently not `adapt`ed correctly
    @test real(semi.mesh) == Float64
    @test typeof(semi.equations.advection_velocity) == SVector{3, Float32}

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

@testitem "KernelAbstractions CPU 3D: elixir_euler_source_terms_nonperiodic.jl native" setup=[
    Setup,
    KernelAbstractionsExamples
] tags=[:kernelabstractions] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "p4est_3d_dgsem",
                                 "elixir_euler_source_terms_nonperiodic.jl"),
                        l2=[0.0014517629881062517,
                            0.0014469623017050836,
                            0.001446962301705153,
                            0.0014469623017051368,
                            0.002934065359862918],
                        linf=[0.01031578086475382,
                            0.011300883615913193,
                            0.011300883615896096,
                            0.011300883615918522,
                            0.02090696711453477],
                        volume_integral=VolumeIntegralFluxDifferencing(flux_kennedy_gruber))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    # The elixir adapts `semi` before creating the ODE problem
    @test semi === ode.p
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

@testitem "KernelAbstractions CPU 3D: boundary flux with an empty boundary condition type" setup=[
    Setup,
    KernelAbstractionsExamples
] tags=[:kernelabstractions] begin
    # A boundary condition type without any boundaries (e.g., on a rank that
    # does not own boundaries of this name in parallel runs) must not stop the
    # processing of the remaining boundary condition types in
    # `calc_boundary_flux_by_type!`.
    trixi_include(@__MODULE__,
                  joinpath(EXAMPLES_DIR, "p4est_3d_dgsem",
                           "elixir_euler_source_terms_nonperiodic.jl"),
                  tspan = (0.0, 0.0))
    semi = ode.p # `semidiscretize` adapts the semi, so we need to obtain it from the ODE problem.
    (; mesh, equations, solver, cache, boundary_conditions) = semi
    @test boundary_conditions isa Trixi.UnstructuredSortedBoundaryTypes
    u_ode = copy(ode.u0)
    u = Trixi.wrap_array(u_ode, semi)
    backend = Trixi.trixi_backend(u_ode)
    t = first(ode.tspan)
    surface_flux_values = cache.elements.surface_flux_values

    # Reference boundary fluxes with the original boundary conditions
    Trixi.prolong2boundaries!(backend, cache, u, mesh, equations, solver)
    fill!(surface_flux_values, 0)
    Trixi.calc_boundary_flux!(backend, cache, t, boundary_conditions, mesh, equations,
                              solver.surface_integral, solver)
    surface_flux_values_reference = copy(surface_flux_values)
    @test any(!iszero, surface_flux_values_reference)

    # Prepend a boundary condition type without any boundaries
    boundary_condition_types = (boundary_condition_do_nothing,
                                boundary_conditions.boundary_condition_types...)
    boundary_indices = (Int[], boundary_conditions.boundary_indices...)
    boundary_conditions_empty_first = Trixi.UnstructuredSortedBoundaryTypes{length(boundary_condition_types),
                                                                            typeof(boundary_condition_types),
                                                                            Vector{Int},
                                                                            typeof(boundary_conditions.boundary_conditions)}(boundary_condition_types,
                                                                                                                             boundary_indices,
                                                                                                                             boundary_conditions.boundary_conditions,
                                                                                                                             boundary_conditions.boundary_symbol_indices)
    fill!(surface_flux_values, 0)
    Trixi.calc_boundary_flux!(backend, cache, t, boundary_conditions_empty_first, mesh,
                              equations, solver.surface_integral, solver)
    @test surface_flux_values == surface_flux_values_reference
end
@testitem "KernelAbstractions CPU 3D: elixir_euler_source_terms_nonperiodic.jl Float32" setup=[
    Setup,
    KernelAbstractionsExamples
] tags=[:kernelabstractions] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "p4est_3d_dgsem",
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
                        real_type=Float32)
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    # The elixir adapts `semi` before creating the ODE problem
    @test semi === ode.p
    @test real(semi.solver) == Float32
    @test real(semi.solver.basis) == Float32
    @test real(semi.solver.mortar) == Float32
    # TODO: `mesh` is currently not `adapt`ed correctly
    @test real(semi.mesh) == Float64
    @test typeof(semi.equations.gamma) == Float32

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

@testitem "KernelAbstractions CPU 3D: elixir_mhd_alfven_wave_combined_fluxes_nonperiodic.jl native" setup=[
    Setup,
    KernelAbstractionsExamples
] tags=[:kernelabstractions] begin
    using Trixi
    @test_trixi_include(joinpath(EXAMPLES_DIR, "p4est_3d_dgsem",
                                 "elixir_mhd_alfven_wave_combined_fluxes_nonperiodic.jl"),
                        l2=[
                            0.00020523607157908708,
                            0.000636037375360857,
                            0.0002768722721598528,
                            0.000780404043245244,
                            0.000709669882361381,
                            0.0006335012491207207,
                            0.0003174622818755188,
                            0.0007723389125485991,
                            0.00023680042414966782
                        ],
                        linf=[
                            0.0011634542712232188,
                            0.004143013256670491,
                            0.0016758874736150725,
                            0.004705809053315052,
                            0.0043527912850840345,
                            0.00389837136357718,
                            0.0014571305070021356,
                            0.00445327133732587,
                            0.0011999511085407923
                        ])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    # The elixir adapts `semi` before creating the ODE problem
    @test semi === ode.p
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

@testitem "KernelAbstractions CPU 3D: elixir_mhd_alfven_wave_combined_fluxes_nonperiodic.jl Float32" setup=[
    Setup,
    KernelAbstractionsExamples
] tags=[:kernelabstractions] begin
    using Trixi
    @test_trixi_include(joinpath(EXAMPLES_DIR, "p4est_3d_dgsem",
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
                        real_type=Float32)
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    # The elixir adapts `semi` before creating the ODE problem
    @test semi === ode.p
    @test real(semi.solver) == Float32
    @test real(semi.solver.basis) == Float32
    @test real(semi.solver.mortar) == Float32
    # TODO: `mesh` is currently not `adapt`ed correctly
    @test real(semi.mesh) == Float64
    @test typeof(semi.equations.gamma) == Float32

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

@testitem "KernelAbstractions CPU 3D: elixir_navierstokes_convergence.jl" setup=[
    Setup,
    KernelAbstractionsExamples
] tags=[:kernelabstractions] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "p4est_3d_dgsem",
                                 "elixir_navierstokes_convergence.jl"),
                        initial_refinement_level=2, tspan=(0.0, 0.1),
                        # Expected errors are exactly the same as in the parabolic test!
                        l2=[
                            0.00026599105557723507,
                            0.00046187779448444603,
                            0.0005424899076194272,
                            0.00046187779448445546,
                            0.0015846392584275121
                        ],
                        linf=[
                            0.0025241668964857134,
                            0.006308461684409397,
                            0.004334939668473314,
                            0.006308461684396753,
                            0.03176343483364796
                        ])
    semi = ode.p # `semidiscretize` adapts the semi, so we need to obtain it from the ODE problem.
    @test real(semi.solver) == Float64
    @test ode.u0 isa Array
    @test Trixi.trixi_backend(ode.u0) isa Trixi.KernelAbstractions.CPU
    @test semi.cache_parabolic.parabolic_container.u_transformed isa Array

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
