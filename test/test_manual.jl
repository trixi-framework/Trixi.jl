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

  @testset "Tree" begin
    @testset "constructors" begin
      @test_nowarn Trixi.Tree(Val(1), 10, 0.0, 1.0)
    end

    @testset "helper functions" begin
      t = Trixi.Tree(Val(1), 10, 0.0, 1.0)
      @test_nowarn show(t)
      @test Trixi.ndims(t) == 1
      @test Trixi.ndims(Trixi.Tree{1}) == 1
      @test Trixi.has_any_neighbor(t, 1, 1) == true
      @test Trixi.isperiodic(t, 1) == true
      @test Trixi.n_children_per_cell(t) == 2
      @test Trixi.n_children_per_cell(2) == 4
      @test Trixi.n_directions(t) == 2
    end

    @testset "refine!/coarsen!" begin
      t = Trixi.Tree(Val(1), 10, 0.0, 1.0)
      @test Trixi.refine!(t) == [1]
      @test Trixi.coarsen!(t) == [1]
      @test Trixi.refine!(t) == [1]
      @test Trixi.coarsen!(t, 1) == [1]
      @test Trixi.coarsen!(t) == Int[] # Coarsen twice to check degenerate case of single-cell tree
      @test Trixi.refine!(t) == [1]
      @test Trixi.refine!(t) == [2,3]
      @test Trixi.coarsen_box!(t, [-0.5], [0.0]) == [2]
      @test Trixi.coarsen_box!(t, 0.0, 0.5) == [3]
      @test isnothing(Trixi.reset_data_structures!(t))
    end
  end
end
