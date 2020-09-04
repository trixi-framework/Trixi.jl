module TestManual

using Test
using Trixi

# Start with a clean environment: remove Trixi output directory if it exists
outdir = "out"
isdir(outdir) && rm(outdir, recursive=true)

# Run various manual (= non-parameter-file-triggered tests)
@testset "Manual tests" begin
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

  @testset "interpolation" begin
    @testset "nodes and weights" begin
      @test Trixi.gauss_nodes_weights(1) == ([0.0], [2.0])
    end

    @testset "multiply_dimensionwise" begin
      nodes_in  = [0.0, 0.5, 1.0]
      nodes_out = [0.0, 1/3, 2/3, 1.0]
      matrix = Trixi.polynomial_interpolation_matrix(nodes_in, nodes_out)
      data_in = [3.0 4.5 6.0]
      @test Trixi.multiply_dimensionwise(matrix, data_in) ≈ [3.0 4.0 5.0 6.0]

      n_vars   = 3
      size_in  = 4
      size_out = 5
      matrix   = randn(size_out, size_in)
      # 1D
      data_in  = randn(n_vars, size_in)
      data_out = Trixi.multiply_dimensionwise_naive(matrix, data_in)
      @test data_out ≈ Trixi.multiply_dimensionwise(matrix, data_in)
      # 2D
      data_in  = randn(n_vars, size_in, size_in)
      data_out = Trixi.multiply_dimensionwise_naive(matrix, data_in)
      @test data_out ≈ Trixi.multiply_dimensionwise(matrix, data_in)
      # 3D
      data_in  = randn(n_vars, size_in, size_in, size_in)
      data_out = Trixi.multiply_dimensionwise_naive(matrix, data_in)
      @test data_out ≈ Trixi.multiply_dimensionwise(matrix, data_in)
    end
  end

  @testset "containers" begin
    # Set up mock container
    mutable struct MyContainer <: Trixi.AbstractContainer
      data::Vector{Int}
      capacity::Int
      length::Int
      dummy::Int
    end
    function MyContainer(data, capacity)
      c = MyContainer(Vector{Int}(undef, capacity+1), capacity, length(data), capacity+1)
      c.data[1:length(data)] .= data
      return c
    end
    MyContainer(data::AbstractArray) = MyContainer(data, length(data))
    Trixi.invalidate!(c::MyContainer, first, last) = (c.data[first:last] .= 0; c)
    function Trixi.raw_copy!(target::MyContainer, source::MyContainer, first, last, destination)
      Trixi.copy_data!(target.data, source.data, first, last, destination)
      return target
    end
    Trixi.move_connectivity!(c::MyContainer, first, last, destination) = c
    Trixi.delete_connectivity!(c::MyContainer, first, last) = c
    Trixi.reset_data_structures!(c::MyContainer) = (c.data = Vector{Int}(undef, c.capacity+1); c)
    function Base.:(==)(c1::MyContainer, c2::MyContainer)
      return (c1.capacity == c2.capacity &&
              c1.length == c2.length &&
              c1.dummy == c2.dummy &&
              c1.data[1:c1.length] == c2.data[1:c2.length])
    end

    @testset "size" begin
      c = MyContainer([1, 2, 3])
      @test size(c) == (3,)
    end

    @testset "resize!" begin
      c = MyContainer([1, 2, 3])
      @test length(resize!(c, 2)) == 2
    end

    @testset "copy!" begin
      c1 = MyContainer([1, 2, 3])
      c2 = MyContainer([4, 5])
      @test Trixi.copy!(c1, c2, 2, 1, 2) == MyContainer([1, 2, 3]) # no-op

      c1 = MyContainer([1, 2, 3])
      c2 = MyContainer([4, 5])
      @test Trixi.copy!(c1, c2, 1, 2, 2) == MyContainer([1, 4, 5])

      c1 = MyContainer([1, 2, 3])
      @test Trixi.copy!(c1, c2, 1, 2) == MyContainer([1, 4, 3])

      c1 = MyContainer([1, 2, 3])
      @test Trixi.copy!(c1, 2, 3, 1) == MyContainer([2, 3, 3])

      c1 = MyContainer([1, 2, 3])
      @test Trixi.copy!(c1, 1, 3) == MyContainer([1, 2, 1])
    end

    @testset "move!" begin
      c = MyContainer([1, 2, 3])
      @test Trixi.move!(c, 1, 1) == MyContainer([1, 2, 3]) # no-op

      c = MyContainer([1, 2, 3])
      @test Trixi.move!(c, 1, 2) == MyContainer([0, 1, 3])
    end

    @testset "swap!" begin
      c = MyContainer([1,2])
      @test Trixi.swap!(c, 1, 1) == MyContainer([1, 2]) # no-op

      c = MyContainer([1,2])
      @test Trixi.swap!(c, 1, 2) == MyContainer([2,1])
    end

    @testset "erase!" begin
      c = MyContainer([1, 2])
      @test Trixi.erase!(c, 2, 1) == MyContainer([1, 2]) # no-op

      c = MyContainer([1, 2])
      @test Trixi.erase!(c, 1) == MyContainer([0, 2])
    end

    @testset "remove_shift!" begin
      c = MyContainer([1, 2, 3, 4])
      @test Trixi.remove_shift!(c, 2, 1) == MyContainer([1, 2, 3, 4]) # no-op

      c = MyContainer([1, 2, 3, 4])
      @test Trixi.remove_shift!(c, 2, 2) == MyContainer([1, 3, 4], 4)

      c = MyContainer([1, 2, 3, 4])
      @test Trixi.remove_shift!(c, 2) == MyContainer([1, 3, 4], 4)
    end

    @testset "remove_fill!" begin
      c = MyContainer([1, 2, 3, 4])
      @test Trixi.remove_fill!(c, 2, 1) == MyContainer([1, 2, 3, 4]) # no-op

      c = MyContainer([1, 2, 3, 4])
      @show "jo"
      @test Trixi.remove_fill!(c, 2, 2) == MyContainer([1, 4, 3], 4)
    end

    @testset "reset!" begin
      c = MyContainer([1, 2, 3])
      @test Trixi.reset!(c, 2) == MyContainer(Int[], 2)
    end
  end

  @testset "example parameters" begin
    @test basename(examples_dir()) == "examples"
    @test !isempty(get_examples())
    @test endswith(default_example(), "parameters.toml")
  end

  @testset "DG L2 mortar container debug output" begin
    c2d = Trixi.L2MortarContainer2D{1, 1}(1)
    @test isnothing(show(c2d))
    c3d = Trixi.L2MortarContainer3D{1, 1}(1)
    @test isnothing(show(c3d))
  end
end

end #module
