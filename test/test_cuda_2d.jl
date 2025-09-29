module TestCUDA

using Test
using Trixi

include("test_trixi.jl")

EXAMPLES_DIR = joinpath(examples_dir(), "p4est_2d_dgsem")

# Start with a clean environment: remove Trixi.jl output directory if it exists
outdir = "out"
isdir(outdir) && rm(outdir, recursive = true)

@testset "CUDA 2D" begin
#! format: noindent

@trixi_testset "elixir_advection_basic_gpu.jl native" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_basic_gpu.jl"),
                        # Expected errors are exactly the same as with TreeMesh!
                        l2=8.311947673061856e-6,
                        linf=6.627000273229378e-5,)
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)
    @test real(ode.p.solver) == Float64
    @test real(ode.p.solver.basis) == Float64
    @test real(ode.p.solver.mortar) == Float64
    # TODO: remake ignores the mesh itself as well
    @test real(ode.p.mesh) == Float64

    @test ode.u0 isa Array
    @test ode.p.solver.basis.derivative_matrix isa Array

    @test Trixi.storage_type(ode.p.cache.elements) === Array
    @test Trixi.storage_type(ode.p.cache.interfaces) === Array
    @test Trixi.storage_type(ode.p.cache.boundaries) === Array
    @test Trixi.storage_type(ode.p.cache.mortars) === Array
end

@trixi_testset "elixir_advection_basic_gpu.jl Float32 / CUDA" begin
    # Using CUDA inside the testset since otherwise the bindings are hiddend by the anonymous modules
    using CUDA
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_basic_gpu.jl"),
                        # Expected errors are exactly the same as with TreeMesh!
                        l2=nothing,   # TODO: GPU. [Float32(8.311947673061856e-6)],
                        linf=nothing, # TODO: GPU. [Float32(6.627000273229378e-5)],
                        RealT=Float32,
                        real_type=Float32,
                        storage_type=CuArray,
                        sol=nothing,) # TODO: GPU. Remove this once we can run the simulation on the GPU
    # # Ensure that we do not have excessive memory allocations
    # # (e.g., from type instabilities)
    # @test_allocations(Trixi.rhs!, semi, sol, 1000)
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

# Clean up afterwards: delete Trixi.jl output directory
@test_nowarn isdir(outdir) && rm(outdir, recursive = true)
end
end # module
