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
    semi = ode.p # `semidiscretize` adapts the semi, so we need to obtain it from the ODE problem.
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
    semi = ode.p # `semidiscretize` adapts the semi, so we need to obtain it from the ODE problem.
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
    semi = ode.p # `semidiscretize` adapts the semi, so we need to obtain it from the ODE problem.
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
    semi = ode.p # `semidiscretize` adapts the semi, so we need to obtain it from the ODE problem.
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
end

@testitem "KernelAbstractions CPU 2D: elixir_mhd_alfven_wave_combined_fluxes_nonperiodic.jl native" setup=[
    Setup,
    KernelAbstractionsExamples
] tags=[:kernelabstractions] begin
    using Trixi
    @test_trixi_include(joinpath(EXAMPLES_DIR, "p4est_2d_dgsem",
                                 "elixir_mhd_alfven_wave_combined_fluxes_nonperiodic.jl"),
                        l2=[
                            8.278171964502251e-5,
                            6.67400550711942e-5,
                            6.693513155020543e-5,
                            0.00011718619995309785,
                            6.889365943089829e-5,
                            7.782210267643806e-5,
                            7.820713512060046e-5,
                            0.00011507076348866596,
                            5.379656409151357e-5
                        ],
                        linf=[
                            0.00042882216116346683,
                            0.000536686629082607,
                            0.0005330550796081301,
                            0.0009163321918530948,
                            0.00042853551496602194,
                            0.0005049089113187133,
                            0.0005058353675793104,
                            0.0008948904521319523,
                            0.00018926467653786568
                        ])
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
end

@testitem "KernelAbstractions CPU 2D: elixir_mhd_alfven_wave_combined_fluxes_nonperiodic.jl Float32" setup=[
    Setup,
    KernelAbstractionsExamples
] tags=[:kernelabstractions] begin
    using Trixi
    @test_trixi_include(joinpath(EXAMPLES_DIR, "p4est_2d_dgsem",
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
                        real_type=Float32)
    semi = ode.p # `semidiscretize` adapts the semi, so we need to obtain it from the ODE problem.
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
end

@testitem "KernelAbstractions CPU 2D: elixir_advection_nonconforming_flag.jl Float32 / CUDA" setup=[
    Setup,
    KernelAbstractionsExamples
] tags=[:kernelabstractions] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR,
                                 "elixir_advection_nonconforming_flag.jl"),
                        l2=Float32[3.198940059144588e-5],
                        linf=Float32[0.00030636069494005547],
                        RealT_for_test_tolerances=Float32,
                        real_type=Float32)
    semi = ode.p # `semidiscretize` adapts the semi, so we need to obtain it from the ODE problem.
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
    semi = ode.p # `semidiscretize` adapts the semi, so we need to obtain it from the ODE problem.
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
    semi = ode.p # `semidiscretize` adapts the semi, so we need to obtain it from the ODE problem.
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
    semi = ode.p # `semidiscretize` adapts the semi, so we need to obtain it from the ODE problem.
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
end

@testitem "KernelAbstractions CPU 3D: elixir_mhd_alfven_wave_combined_fluxes_nonperiodic.jl native" setup=[
    Setup,
    KernelAbstractionsExamples
] tags=[:kernelabstractions] begin
    using Trixi
    @test_trixi_include(joinpath(EXAMPLES_DIR, "p4est_3d_dgsem",
                                 "elixir_mhd_alfven_wave_combined_fluxes_nonperiodic.jl"),
                        l2=[
                            0.00021050235921250785,
                            0.0006558863249658414,
                            0.0002821364462491609,
                            0.000794748439799794,
                            0.0006839039331448021,
                            0.0006743445567763623,
                            0.00031815692647892813,
                            0.0007885451813871558,
                            4.811726181476006e-5
                        ],
                        linf=[
                            0.0012031070458876636,
                            0.00410699976203599,
                            0.0017830978311310533,
                            0.004780625099412877,
                            0.0050959023689367555,
                            0.003922455896960386,
                            0.002515549812865392,
                            0.004448527707559019,
                            0.0001983994478820785
                        ])
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
end

@testitem "KernelAbstractions CPU 3D: elixir_mhd_alfven_wave_combined_fluxes_nonperiodic.jl Float32" setup=[
    Setup,
    KernelAbstractionsExamples
] tags=[:kernelabstractions] begin
    using Trixi
    @test_trixi_include(joinpath(EXAMPLES_DIR, "p4est_3d_dgsem",
                                 "elixir_mhd_alfven_wave_combined_fluxes_nonperiodic.jl"),
                        l2=Float32[0.00021050235826592327,
                                   0.0006558863204839041,
                                   0.0002821364444400733,
                                   0.000794748435433683,
                                   0.0006839039307848098,
                                   0.0006743445524692008,
                                   0.000318156924452865,
                                   0.0007885451771559438,
                                   4.811726173404515e-5],
                        linf=Float32[0.0012031070350810857,
                                     0.004106999758487398,
                                     0.001783097816025008,
                                     0.004780625055122056,
                                     0.005095902318184908,
                                     0.003922455893839549,
                                     0.002515549802432071,
                                     0.004448527671538249,
                                     0.00019839944646198146],
                        RealT_for_test_tolerances=Float32,
                        real_type=Float32)
    semi = ode.p # `semidiscretize` adapts the semi, so we need to obtain it from the ODE problem.
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
end
