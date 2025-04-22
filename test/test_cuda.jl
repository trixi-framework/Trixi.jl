module TestCUDA

using CUDA
using Test
using Trixi

include("test_trixi.jl")

# EXAMPLES_DIR = joinpath(examples_dir(), "dgmulti_1d")

# Start with a clean environment: remove Trixi.jl output directory if it exists
outdir = "out"
isdir(outdir) && rm(outdir, recursive = true)

EXAMPLES_DIR = joinpath(examples_dir(), "p4est_2d_dgsem")

@trixi_testset "elixir_advection_basic.jl (Float32)" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_basic.jl"),
                        # Expected errors are exactly the same as with TreeMesh!
                        l2=[8.311947673061856e-6],
                        linf=[6.627000273229378e-5],
                        real_type=Float32,
                        storage_type=CuArray)
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
end

# Clean up afterwards: delete Trixi.jl output directory
@test_nowarn isdir(outdir) && rm(outdir, recursive = true)

end # module
