@testsnippet KernelAbstractionsMPIExamples begin
    EXAMPLES_DIR = examples_dir()
end

@testitem "KernelAbstractions MPI backend preference" setup=[Setup] tags=[:kernelabstractions_mpi] begin
    @test Trixi._PREFERENCE_THREADING == :kernelabstractions
end

@testitem "KernelAbstractions MPI CPU 2D: elixir_advection_basic.jl" setup=[
    Setup,
    KernelAbstractionsMPIExamples
] tags=[:kernelabstractions_mpi] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "p4est_2d_dgsem",
                                 "elixir_advection_basic.jl"),
                        # Expected errors are exactly the same as with TreeMesh!
                        l2=8.311947673061856e-6,
                        linf=6.627000273229378e-5)
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
    Trixi.rhs_hyperbolic!(du_ode, u_ode, semi, first(ode.tspan))
    @test all(isfinite, du_ode)
end

# The unstructured mesh contains MPI interfaces with flipped orientations
@testitem "KernelAbstractions MPI CPU 2D: elixir_advection_unstructured_flag.jl" setup=[
    Setup,
    KernelAbstractionsMPIExamples
] tags=[:kernelabstractions_mpi] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "p4est_2d_dgsem",
                                 "elixir_advection_unstructured_flag.jl"),
                        # Expected errors are exactly the same as in the CPU version
                        l2=[0.0005379687442422346],
                        linf=[0.007438525029884735])
end

@testitem "KernelAbstractions MPI CPU 2D: elixir_mhd_alfven_wave_combined_fluxes_nonperiodic.jl" setup=[
    Setup,
    KernelAbstractionsMPIExamples
] tags=[:kernelabstractions_mpi] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "p4est_2d_dgsem",
                                 "elixir_mhd_alfven_wave_combined_fluxes_nonperiodic.jl"),
                        # Expected errors agree with the non-MPI version up to round-off
                        l2=[
                            7.475384025292345e-5,
                            6.508061190791895e-5,
                            6.494699189562195e-5,
                            0.000100398590022096,
                            9.533883559805393e-5,
                            7.594652450345156e-5,
                            7.599093148092314e-5,
                            9.723421960617558e-5,
                            5.289168299208382e-5
                        ],
                        linf=[
                            0.0003427579185457663,
                            0.0004998303440423985,
                            0.0004839939536228563,
                            0.0005180908760881453,
                            0.0004278890186213413,
                            0.0004136795102114288,
                            0.0004278210815760186,
                            0.0005252948274238056,
                            0.00015843034878747502
                        ],
                        tspan=(0.0, 0.1))
end

@testitem "KernelAbstractions MPI CPU 3D: elixir_advection_basic.jl" setup=[
    Setup,
    KernelAbstractionsMPIExamples
] tags=[:kernelabstractions_mpi] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "p4est_3d_dgsem",
                                 "elixir_advection_basic.jl"),
                        # Expected errors are exactly the same as with TreeMesh!
                        l2=[0.00016263963870641478],
                        linf=[0.0014537194925779984])
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
    Trixi.rhs_hyperbolic!(du_ode, u_ode, semi, first(ode.tspan))
    @test all(isfinite, du_ode)
end

# The unstructured mesh contains MPI interfaces with flipped orientations
@testitem "KernelAbstractions MPI CPU 3D: elixir_advection_unstructured_curved.jl" setup=[Setup, KernelAbstractionsMPIExamples] tags=[:kernelabstractions_mpi] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "p4est_3d_dgsem",
                                 "elixir_advection_unstructured_curved.jl"),
                        # Expected errors agree with the non-MPI version up to round-off
                        l2=[0.0004750004258546538],
                        linf=[0.026527551737137167])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
end

@testitem "KernelAbstractions MPI CPU 3D: elixir_mhd_alfven_wave_combined_fluxes_nonperiodic.jl native" setup=[
    Setup,
    KernelAbstractionsMPIExamples
] tags=[:kernelabstractions_mpi] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR,"p4est_3d_dgsem",
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
                        ],
                        tspan=(0.0, 0.1))
end
