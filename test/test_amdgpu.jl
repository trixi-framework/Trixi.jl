module TestAMDGPU

using Test
using Trixi

include("test_trixi.jl")

# Start with a clean environment: remove Trixi.jl output directory if it exists
outdir = "out"
isdir(outdir) && rm(outdir, recursive = true)

EXAMPLES_DIR = joinpath(examples_dir(), "p4est_2d_dgsem")

@trixi_testset "elixir_advection_basic_gpu.jl native" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_basic_gpu.jl"),
                        # Expected errors are exactly the same as with TreeMesh!
                        l2=nothing,   # [Float32(8.311947673061856e-6)],
                        linf=nothing,)
    # # Ensure that we do not have excessive memory allocations
    # # (e.g., from type instabilities)
    # let
    #     t = sol.t[end]
    #     u_ode = sol.u[end]
    #     du_ode = similar(u_ode)
    #     @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    # end
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

@trixi_testset "elixir_advection_basic_gpu.jl Float32 / AMDGPU" begin
    # Using AMDGPU inside the testset since otherwise the bindings are hiddend by the anonymous modules
    using AMDGPU
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_basic_gpu.jl"),
                        # Expected errors are exactly the same as with TreeMesh!
                        l2=nothing,   # [Float32(8.311947673061856e-6)],
                        linf=nothing, # [Float32(6.627000273229378e-5)],
                        real_type=Float32,
                        storage_type=ROCArray)
    # # Ensure that we do not have excessive memory allocations
    # # (e.g., from type instabilities)
    # let
    #     t = sol.t[end]
    #     u_ode = sol.u[end]
    #     du_ode = similar(u_ode)
    #     @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    # end
    @test real(ode.p.solver) == Float32
    @test real(ode.p.solver.basis) == Float32
    @test real(ode.p.solver.mortar) == Float32
    # TODO: remake ignores the mesh itself as well
    @test real(ode.p.mesh) == Float64

    @test ode.u0 isa ROCArray
    @test ode.p.solver.basis.derivative_matrix isa ROCArray

    @test Trixi.storage_type(ode.p.cache.elements) === ROCArray
    @test Trixi.storage_type(ode.p.cache.interfaces) === ROCArray
    @test Trixi.storage_type(ode.p.cache.boundaries) === ROCArray
    @test Trixi.storage_type(ode.p.cache.mortars) === ROCArray
end

# Clean up afterwards: delete Trixi.jl output directory
@test_nowarn isdir(outdir) && rm(outdir, recursive = true)

end # module
