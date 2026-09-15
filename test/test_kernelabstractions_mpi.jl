@testsnippet KernelAbstractionsMPIExamples begin
    EXAMPLES_DIR = examples_dir()
end

@testitem "KernelAbstractions MPI backend preference" setup=[Setup] tags=[:kernelabstractions_mpi] begin
    @test Trixi._PREFERENCE_THREADING == :kernelabstractions
end

@testitem "KernelAbstractions MPI GPU 2D: elixir_advection_basic.jl" setup=[
    Setup,
    KernelAbstractionsMPIExamples
] tags=[:kernelabstractions_mpi] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "p4est_2d_dgsem",
                                 "elixir_advection_basic.jl"),
                        # Expected errors are exactly the same as with TreeMesh!
                        l2=8.311947673061856e-6,
                        linf=6.627000273229378e-5)
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
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

    # Ensure that the RHS computation overwrites existing data in `du` correctly.
    u_ode = copy(ode.u0)
    du_ode = similar(u_ode)
    fill!(du_ode, convert(eltype(du_ode), NaN))
    Trixi.rhs_hyperbolic!(du_ode, u_ode, ode.p, first(ode.tspan))
    @test all(isfinite, du_ode)
end
