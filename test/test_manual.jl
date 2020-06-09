using Test
import Trixi

# Start with a clean environment: remove Trixi output directory if it exists
outdir = joinpath(@__DIR__, "out")
isdir(outdir) && rm(outdir, recursive=true)

# Run various manual (= non-parameter-file-triggered tests)
@testset "Manual tests" begin
  @testset "parse_commandline_arguments" begin
    args = ["-h"]
    @test Trixi.parse_commandline_arguments(args, testing=true) == 1
    args = ["-"]
    @test Trixi.parse_commandline_arguments(args, testing=true) == 2
    args = ["filename", "filename"]
    @test Trixi.parse_commandline_arguments(args, testing=true) == 3
    args = ["-v"]
    @test Trixi.parse_commandline_arguments(args, testing=true) == 4
    args = ["filename"]
    @test_nowarn Trixi.parse_commandline_arguments(args, testing=true)
  end
end
