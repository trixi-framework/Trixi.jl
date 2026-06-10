module TestCUDA3D

using Test
using Trixi

include("test_trixi.jl")

EXAMPLES_DIR = joinpath(examples_dir(), "p4est_3d_dgsem")

# Start with a clean environment: remove Trixi.jl output directory if it exists
outdir = "out"
isdir(outdir) && rm(outdir, recursive = true)

@testset "CUDA 3D" begin
#! format: noindent

@trixi_testset "elixir_advection_basic.jl native" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_basic.jl"),
                        # Expected errors are exactly the same as with TreeMesh!
                        l2=[0.00016263963870641478],
                        linf=[0.0014537194925779984])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    semi = ode.p # `semidiscretize` adapts the semi, so we need to obtain it from the ODE problem.
    @test_allocations(Trixi.rhs!, semi, sol, 1000)
    @test real(ode.p.solver) == Float64
    @test real(ode.p.solver.basis) == Float64
    @test real(ode.p.solver.mortar) == Float64
    # TODO: remake ignores the mesh itself as well
    @test real(ode.p.mesh) == Float64

    @test ode.u0 isa TrixiStateVector
    @test ode.p.solver.basis.derivative_matrix isa Array

    @test Trixi.storage_type(ode.p.cache.elements) === Array
    @test Trixi.storage_type(ode.p.cache.interfaces) === Array
    @test Trixi.storage_type(ode.p.cache.boundaries) === Array
    @test Trixi.storage_type(ode.p.cache.mortars) === Array
end

@trixi_testset "elixir_advection_basic.jl Float32 / CUDA" begin
    # Using CUDA inside the testset since otherwise the bindings are hiddend by the anonymous modules
    using CUDA
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_basic.jl"),
                        # Expected errors similar to reference on CPU
                        l2=[Float32(0.00016263963870641478)],
                        linf=[Float32(0.0014537194925779984)],
                        RealT_for_test_tolerances=Float32,
                        real_type=Float32,
                        storage_type=CuArray)
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    semi = ode.p # `semidiscretize` adapts the semi, so we need to obtain it from the ODE problem.
    @test_allocations(Trixi.rhs!, semi, sol, 1_700_000)
    @test real(ode.p.solver) == Float32
    @test real(ode.p.solver.basis) == Float32
    @test real(ode.p.solver.mortar) == Float32
    # TODO: remake ignores the mesh itself as well
    @test real(ode.p.mesh) == Float64

    @test ode.u0 isa CuArray
    @test ode.p.solver.basis.derivative_matrix isa CuArray

    @test Trixi.storage_type(ode.p.cache.elements) === CuArray
    @test Trixi.storage_type(ode.p.cache.interfaces) === CuArray
    @test Trixi.storage_type(ode.p.cache.boundaries) === CuArray
    @test Trixi.storage_type(ode.p.cache.mortars) === CuArray
end

@trixi_testset "elixir_euler_source_terms.jl native" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_source_terms.jl"),
                        l2=[
                            4.893619139889976e-5,
                            5.3526950567182756e-5,
                            5.35269505672133e-5,
                            5.352695056735998e-5,
                            0.00015172095200428318
                        ],
                        linf=[
                            0.00031179856625374036,
                            0.0003368725355339386,
                            0.0003368725355383795,
                            0.00033687253560787944,
                            0.0013193387520935573
                        ])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    semi = ode.p # `semidiscretize` adapts the semi, so we need to obtain it from the ODE problem.
    @test_allocations(Trixi.rhs!, semi, sol, 1000)
    @test real(ode.p.solver) == Float64
    @test real(ode.p.solver.basis) == Float64
    @test real(ode.p.solver.mortar) == Float64
    # TODO: `mesh` is currently not `adapt`ed correctly
    @test real(ode.p.mesh) == Float64
    @test typeof(equations.gamma) == Float64

    @test ode.u0 isa Array
    @test ode.p.solver.basis.derivative_matrix isa Array

    @test Trixi.storage_type(ode.p.cache.elements) === Array
    @test Trixi.storage_type(ode.p.cache.interfaces) === Array
    @test Trixi.storage_type(ode.p.cache.boundaries) === Array
    @test Trixi.storage_type(ode.p.cache.mortars) === Array
end

@trixi_testset "elixir_euler_source_terms.jl Float32 / CUDA" begin
    # Using CUDA inside the testset since otherwise the bindings are hiddend by the anonymous modules
    using CUDA
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_source_terms.jl"),
                        l2=Float32[4.912578089985958e-5,
                                   5.3683407014580115e-5,
                                   5.368099834769191e-5,
                                   5.371664525206341e-5,
                                   0.00015186256300882088],
                        linf=Float32[0.00032772542853032327,
                                     0.00035144807715092874,
                                     0.0003549051465479014,
                                     0.00035573961157475686,
                                     0.0013591384887696734],
                        RealT_for_test_tolerances=Float32,
                        real_type=Float32,
                        storage_type=CuArray,
                        gamma=Float32(1.4)) # TODO: This should not be required
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    semi = ode.p # `semidiscretize` adapts the semi, so we need to obtain it from the ODE problem.
    @test_allocations(Trixi.rhs!, semi, sol, 100_000)
    @test real(ode.p.solver) == Float32
    @test real(ode.p.solver.basis) == Float32
    @test real(ode.p.solver.mortar) == Float32
    # TODO: `mesh` is currently not `adapt`ed correctly
    @test real(ode.p.mesh) == Float64
    @test typeof(equations.gamma) == Float32

    @test ode.u0 isa CuArray
    @test ode.p.solver.basis.derivative_matrix isa CuArray

    @test Trixi.storage_type(ode.p.cache.elements) === CuArray
    @test Trixi.storage_type(ode.p.cache.interfaces) === CuArray
    @test Trixi.storage_type(ode.p.cache.boundaries) === CuArray
    @test Trixi.storage_type(ode.p.cache.mortars) === CuArray
end

# Clean up afterwards: delete Trixi.jl output directory
@test_nowarn isdir(outdir) && rm(outdir, recursive = true)
end
end # module
