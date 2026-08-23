module TestExamplesSerial

using Test
using Trixi

include("test_trixi.jl")

EXAMPLES_DIR = examples_dir()

# Start with a clean environment: remove Trixi.jl output directory if it exists
outdir = "out"
Trixi.mpi_isroot() && isdir(outdir) && rm(outdir, recursive = true)
Trixi.MPI.Barrier(Trixi.mpi_comm())

@testset "basic" begin
    @test Trixi._PREFERENCE_THREADING == :serial
end

@testset "Serial 2D" begin
#! format: noindent

@trixi_testset "elixir_advection_amr_refine_twice.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "tree_2d_dgsem",
                                 "elixir_advection_amr_refine_twice.jl"),
                        l2=[0.00020547512522578292],
                        linf=[0.007831753383083506])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 5000)
end

@trixi_testset "elixir_euler_source_terms_nonperiodic.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "tree_2d_dgsem",
                                 "elixir_euler_source_terms_nonperiodic.jl"),
                        l2=[
                            2.259440511766445e-6,
                            2.318888155713922e-6,
                            2.3188881557894307e-6,
                            6.3327863238858925e-6
                        ],
                        linf=[
                            1.498738264560373e-5,
                            1.9182011928187137e-5,
                            1.918201192685487e-5,
                            6.0526717141407005e-5
                        ])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 5000)
end

end # testset

# Clean up afterwards: delete Trixi.jl output directory
Trixi.mpi_isroot() && isdir(outdir) && @test_nowarn rm(outdir, recursive = true)
Trixi.MPI.Barrier(Trixi.mpi_comm())

end # module
