module TestExamplesKernelAbstractions

using Test
using Trixi

include("test_trixi.jl")

EXAMPLES_DIR = examples_dir()

# Start with a clean environment: remove Trixi.jl output directory if it exists
outdir = "out"
Trixi.mpi_isroot() && isdir(outdir) && rm(outdir, recursive = true)
Trixi.MPI.Barrier(Trixi.mpi_comm())

@testset "basic" begin
    @test Trixi._PREFERENCE_THREADING == :kernelabstractions
end

@testset "KernelAbstractions CPU 2D" begin
#! format: noindent

@trixi_testset "elixir_advection_basic_gpu.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "p4est_2d_dgsem",
                                 "elixir_advection_basic_gpu.jl"),
                        # Expected errors are exactly the same as with TreeMesh!
                        l2=8.311947673061856e-6,
                        linf=6.627000273229378e-5)
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    semi = ode.p # `semidiscretize` adapts the semi, so we need to obtain it from the ODE problem.
    @test_allocations(Trixi.rhs!, ode.p, sol, 75_000)
end

@trixi_testset "elixir_advection_basic_gpu.jl Float32" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "p4est_2d_dgsem",
                                 "elixir_advection_basic_gpu.jl"),
                        # Expected errors similar to reference on CPU
                        l2=[Float32(8.311947673061856e-6)],
                        linf=[Float32(6.627000273229378e-5)],
                        RealT_for_test_tolerances=Float32,
                        real_type=Float32)
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    semi = ode.p # `semidiscretize` adapts the semi, so we need to obtain it from the ODE problem.
    @test_allocations(Trixi.rhs!, ode.p, sol, 60_000)
end

@trixi_testset "elixir_euler_source_terms.jl native" begin
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
    semi = ode.p # `semidiscretize` adapts the semi, so we need to obtain it from the ODE problem.
    @test_allocations(Trixi.rhs!, semi, sol, 100_000)
end

@trixi_testset "elixir_euler_source_terms.jl Float32 / AMDGPU" begin
    # Using AMDGPU inside the testset since otherwise the bindings are hiddend by the anonymous modules
    using AMDGPU
    @test_trixi_include(joinpath(EXAMPLES_DIR, "p4est_2d_dgsem",
                                 "elixir_euler_source_terms.jl"),
                        # Expected errors are exactly the same as with TreeMesh!
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
                        gamma=Float32(1.4))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    semi = ode.p # `semidiscretize` adapts the semi, so we need to obtain it from the ODE problem.
    @test_allocations(Trixi.rhs!, semi, sol, 100_000)
end
end

@testset "KernelAbstractions CPU 3D" begin
#! format: noindent

@trixi_testset "elixir_advection_basic_gpu.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "p4est_3d_dgsem",
                                 "elixir_advection_basic_gpu.jl"),
                        # Expected errors are exactly the same as with TreeMesh!
                        l2=[0.00016263963870641478],
                        linf=[0.0014537194925779984])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    semi = ode.p # `semidiscretize` adapts the semi, so we need to obtain it from the ODE problem.
    @test_allocations(Trixi.rhs!, semi, sol, 450_000)
end

@trixi_testset "elixir_advection_basic_gpu.jl Float32" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "p4est_3d_dgsem",
                                 "elixir_advection_basic_gpu.jl"),
                        # Expected errors similar to reference on CPU
                        l2=[Float32(0.00016263963870641478)],
                        linf=[Float32(0.0014537194925779984)],
                        RealT_for_test_tolerances=Float32,
                        real_type=Float32)
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    semi = ode.p # `semidiscretize` adapts the semi, so we need to obtain it from the ODE problem.
    @test_allocations(Trixi.rhs!, semi, sol, 370_000)
end

@trixi_testset "elixir_euler_source_terms.jl native" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "p4est_3d_dgsem",
                                 "elixir_euler_source_terms.jl"),
                        # Expected errors are exactly the same as with TreeMesh!
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
    @test_allocations(Trixi.rhs!, semi, sol, 400_000)
end

@trixi_testset "elixir_euler_source_terms.jl Float32 / AMDGPU" begin
    # Using AMDGPU inside the testset since otherwise the bindings are hiddend by the anonymous modules
    using AMDGPU
    @test_trixi_include(joinpath(EXAMPLES_DIR, "p4est_3d_dgsem",
                                 "elixir_euler_source_terms.jl"),
                        # Expected errors are exactly the same as with TreeMesh!
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
                        gamma=Float32(1.4))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    semi = ode.p # `semidiscretize` adapts the semi, so we need to obtain it from the ODE problem.
    @test_allocations(Trixi.rhs!, semi, sol, 100_000)
end
end

# Clean up afterwards: delete Trixi.jl output directory
Trixi.mpi_isroot() && isdir(outdir) && @test_nowarn rm(outdir, recursive = true)
Trixi.MPI.Barrier(Trixi.mpi_comm())

end # module
