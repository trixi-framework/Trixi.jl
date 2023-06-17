module TestUnit

using Test
using Trixi

include("test_trixi.jl")

# Start with a clean environment: remove Trixi.jl output directory if it exists
outdir = "out"
isdir(outdir) && rm(outdir, recursive=true)

# Run various unit (= non-elixir-triggered) tests
@testset "Unit tests" begin
  @timed_testset "SerialTree" begin
    @testset "constructors" begin
      @test_nowarn Trixi.SerialTree(Val(1), 10, 0.0, 1.0)
    end

    @testset "helper functions" begin
      t = Trixi.SerialTree(Val(1), 10, 0.0, 1.0)
      @test_nowarn display(t)
      @test Trixi.ndims(t) == 1
      @test Trixi.has_any_neighbor(t, 1, 1) == true
      @test Trixi.isperiodic(t, 1) == true
      @test Trixi.n_children_per_cell(t) == 2
      @test Trixi.n_directions(t) == 2
    end

    @testset "refine!/coarsen!" begin
      t = Trixi.SerialTree(Val(1), 10, 0.0, 1.0)
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

  @timed_testset "ParallelTree" begin
    @testset "constructors" begin
      @test_nowarn Trixi.ParallelTree(Val(1), 10, 0.0, 1.0)
    end

    @testset "helper functions" begin
      t = Trixi.ParallelTree(Val(1), 10, 0.0, 1.0)
      @test isnothing(display(t))
      @test isnothing(Trixi.reset_data_structures!(t))
    end
  end

  @timed_testset "TreeMesh" begin
    @testset "constructors" begin
      @test TreeMesh{1, Trixi.SerialTree{1}}(1, 5.0, 2.0) isa TreeMesh
    end
  end

  @timed_testset "ParallelTreeMesh" begin
    @testset "partition!" begin
      @testset "mpi_nranks() = 2" begin
        Trixi.mpi_nranks() = 2
        let
          @test Trixi.mpi_nranks() == 2

          mesh = TreeMesh{2, Trixi.ParallelTree{2}}(30, (0.0, 0.0), 1)
          # Refine twice
          Trixi.refine!(mesh.tree)
          Trixi.refine!(mesh.tree)

          # allow_coarsening = true
          Trixi.partition!(mesh)
          # Use parent for OffsetArray
          @test parent(mesh.n_cells_by_rank) == [11, 10]
          @test mesh.tree.mpi_ranks[1:21] ==
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
          @test parent(mesh.first_cell_by_rank) == [1, 12]

          # allow_coarsening = false
          Trixi.partition!(mesh; allow_coarsening=false)
          @test parent(mesh.n_cells_by_rank) == [11, 10]
          @test mesh.tree.mpi_ranks[1:21] ==
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
          @test parent(mesh.first_cell_by_rank) == [1, 12]
        end
        Trixi.mpi_nranks() = Trixi.MPI_SIZE[] # restore the original behavior
      end

      @testset "mpi_nranks() = 3" begin
        Trixi.mpi_nranks() = 3
        let
          @test Trixi.mpi_nranks() == 3

          mesh = TreeMesh{2, Trixi.ParallelTree{2}}(100, (0.0, 0.0), 1)
          # Refine twice
          Trixi.refine!(mesh.tree)
          Trixi.refine!(mesh.tree)

          # allow_coarsening = true
          Trixi.partition!(mesh)
          # Use parent for OffsetArray
          @test parent(mesh.n_cells_by_rank) == [11, 5, 5]
          @test mesh.tree.mpi_ranks[1:21] ==
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2]
          @test parent(mesh.first_cell_by_rank) == [1, 12, 17]

          # allow_coarsening = false
          Trixi.partition!(mesh; allow_coarsening=false)
          @test parent(mesh.n_cells_by_rank) == [9, 6, 6]
          @test mesh.tree.mpi_ranks[1:21] ==
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2]
          @test parent(mesh.first_cell_by_rank) == [1, 10, 16]
        end
        Trixi.mpi_nranks() = Trixi.MPI_SIZE[] # restore the original behavior
      end

      @testset "mpi_nranks() = 9" begin
        Trixi.mpi_nranks() = 9
        let
          @test Trixi.mpi_nranks() == 9

          mesh = TreeMesh{2, Trixi.ParallelTree{2}}(1000, (0.0, 0.0), 1)
          # Refine twice
          Trixi.refine!(mesh.tree)
          Trixi.refine!(mesh.tree)
          Trixi.refine!(mesh.tree)
          Trixi.refine!(mesh.tree)

          # allow_coarsening = true
          Trixi.partition!(mesh)
          # Use parent for OffsetArray
          @test parent(mesh.n_cells_by_rank) == [44, 37, 38, 37, 37, 37, 38, 37, 36]
          @test parent(mesh.first_cell_by_rank) == [1, 45, 82, 120, 157, 194, 231, 269, 306]
        end
        Trixi.mpi_nranks() = Trixi.MPI_SIZE[] # restore the original behavior
      end

      @testset "mpi_nranks() = 3 non-uniform" begin
        Trixi.mpi_nranks() = 3
        let
          @test Trixi.mpi_nranks() == 3

          mesh = TreeMesh{2, Trixi.ParallelTree{2}}(100, (0.0, 0.0), 1)
          # Refine whole tree
          Trixi.refine!(mesh.tree)
          # Refine left leaf
          Trixi.refine!(mesh.tree, [2])

          # allow_coarsening = true
          Trixi.partition!(mesh)
          # Use parent for OffsetArray
          @test parent(mesh.n_cells_by_rank) == [6, 1, 2]
          @test mesh.tree.mpi_ranks[1:9] == [0, 0, 0, 0, 0, 0, 1, 2, 2]
          @test parent(mesh.first_cell_by_rank) == [1, 7, 8]

          # allow_coarsening = false
          Trixi.partition!(mesh; allow_coarsening=false)
          @test parent(mesh.n_cells_by_rank) == [5, 2, 2]
          @test mesh.tree.mpi_ranks[1:9] == [0, 0, 0, 0, 0, 1, 1, 2, 2]
          @test parent(mesh.first_cell_by_rank) == [1, 6, 8]
        end
        Trixi.mpi_nranks() = Trixi.MPI_SIZE[] # restore the original behavior
      end

      @testset "not enough ranks" begin
        Trixi.mpi_nranks() = 3
        let
          @test Trixi.mpi_nranks() == 3

          mesh = TreeMesh{2, Trixi.ParallelTree{2}}(100, (0.0, 0.0), 1)

          # Only one leaf
          @test_throws AssertionError(
            "Too many ranks to properly partition the mesh!") Trixi.partition!(mesh)

          # Refine to 4 leaves
          Trixi.refine!(mesh.tree)

          # All four leaves will need to be on one rank to allow coarsening
          @test_throws AssertionError(
            "Too many ranks to properly partition the mesh!") Trixi.partition!(mesh)
          @test_nowarn Trixi.partition!(mesh; allow_coarsening=false)
        end
        Trixi.mpi_nranks() = Trixi.MPI_SIZE[] # restore the original behavior
      end
    end
  end

  @timed_testset "curved mesh" begin
    @testset "calc_jacobian_matrix" begin
      @testset "identity map" begin
        basis = LobattoLegendreBasis(5)
        nodes = Trixi.get_nodes(basis)
        jacobian_matrix = Array{Float64, 5}(undef, 2, 2, 6, 6, 1)

        node_coordinates = Array{Float64, 4}(undef, 2, 6, 6, 1)
        node_coordinates[1, :, :, 1] .= [nodes[i] for i in 1:6, j in 1:6]
        node_coordinates[2, :, :, 1] .= [nodes[j] for i in 1:6, j in 1:6]
        expected = zeros(2, 2, 6, 6, 1)
        expected[1, 1, :, :, 1] .= 1
        expected[2, 2, :, :, 1] .= 1
        @test Trixi.calc_jacobian_matrix!(jacobian_matrix, 1, node_coordinates, basis) ≈ expected
      end

      @testset "maximum exact polydeg" begin
        basis = LobattoLegendreBasis(3)
        nodes = Trixi.get_nodes(basis)
        jacobian_matrix = Array{Float64, 5}(undef, 2, 2, 4, 4, 1)

        # f(x, y) = [x^3, xy^2]
        node_coordinates = Array{Float64, 4}(undef, 2, 4, 4, 1)
        node_coordinates[1, :, :, 1] .= [nodes[i]^3 for i in 1:4, j in 1:4]
        node_coordinates[2, :, :, 1] .= [nodes[i] * nodes[j]^2 for i in 1:4, j in 1:4]

        # Df(x, y) = [3x^2 0;
        #              y^2 2xy]
        expected = zeros(2, 2, 4, 4, 1)
        expected[1, 1, :, :, 1] .= [3 * nodes[i]^2 for i in 1:4, j in 1:4]
        expected[2, 1, :, :, 1] .= [nodes[j]^2 for i in 1:4, j in 1:4]
        expected[2, 2, :, :, 1] .= [2 * nodes[i] * nodes[j] for i in 1:4, j in 1:4]
        @test Trixi.calc_jacobian_matrix!(jacobian_matrix, 1, node_coordinates, basis) ≈ expected
      end
    end
  end

  @timed_testset "interpolation" begin
    @testset "nodes and weights" begin
      @test Trixi.gauss_nodes_weights(1) == ([0.0], [2.0])
    end

    @testset "multiply_dimensionwise" begin
      nodes_in  = [0.0, 0.5, 1.0]
      nodes_out = [0.0, 1/3, 2/3, 1.0]
      matrix = Trixi.polynomial_interpolation_matrix(nodes_in, nodes_out)
      data_in = [3.0 4.5 6.0]
      @test isapprox(Trixi.multiply_dimensionwise(matrix, data_in), [3.0 4.0 5.0 6.0])

      n_vars   = 3
      size_in  = 2
      size_out = 3
      matrix   = randn(size_out, size_in)
      # 1D
      data_in  = randn(n_vars, size_in)
      data_out = Trixi.multiply_dimensionwise_naive(matrix, data_in)
      @test isapprox(data_out, Trixi.multiply_dimensionwise(matrix, data_in))
      # 2D
      data_in  = randn(n_vars, size_in, size_in)
      data_out = Trixi.multiply_dimensionwise_naive(matrix, data_in)
      @test isapprox(data_out, Trixi.multiply_dimensionwise(matrix, data_in))
      # 3D
      data_in  = randn(n_vars, size_in, size_in, size_in)
      data_out = Trixi.multiply_dimensionwise_naive(matrix, data_in)
      @test isapprox(data_out, Trixi.multiply_dimensionwise(matrix, data_in))
    end
  end

  @timed_testset "L2 projection" begin
    @testset "calc_reverse_upper for LGL" begin
      @test isapprox(Trixi.calc_reverse_upper(2, Val(:gauss_lobatto)), [[0.25, 0.25] [0.0, 0.5]])
    end
    @testset "calc_reverse_lower for LGL" begin
      @test isapprox(Trixi.calc_reverse_lower(2, Val(:gauss_lobatto)), [[0.5, 0.0] [0.25, 0.25]])
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
      @test Trixi.remove_fill!(c, 2, 2) == MyContainer([1, 4, 3], 4)
    end

    @testset "reset!" begin
      c = MyContainer([1, 2, 3])
      @test Trixi.reset!(c, 2) == MyContainer(Int[], 2)
    end
  end

  @timed_testset "example elixirs" begin
    @test basename(examples_dir()) == "examples"
    @test !isempty(get_examples())
    @test endswith(default_example(), "elixir_advection_basic.jl")
  end

  @timed_testset "HLL flux with vanishing wave speed estimates (#502)" begin
    equations = CompressibleEulerEquations1D(1.4)
    u = SVector(1.0, 0.0, 0.0)
    @test !any(isnan, FluxHLL()(u, u, 1, equations))
  end

  @timed_testset "DG L2 mortar container debug output" begin
    c2d = Trixi.L2MortarContainer2D{Float64}(1, 1, 1)
    @test isnothing(display(c2d))
    c3d = Trixi.L2MortarContainer3D{Float64}(1, 1, 1)
    @test isnothing(display(c3d))
  end

  @timed_testset "Printing indicators/controllers" begin
    # OBS! Constructing indicators/controllers using the parameters below doesn't make sense. It's
    # just useful to run basic tests of `show` methods.

    c = ControllerThreeLevelCombined(1, 2, 3, 10.0, 11.0, 12.0, "primary", "secondary", "cache")
    @test_nowarn show(stdout, c)

    indicator_hg = IndicatorHennemannGassner(1.0, 0.0, true, "variable", "cache")
    @test_nowarn show(stdout, indicator_hg)

    indicator_loehner = IndicatorLöhner(1.0, "variable", (; cache=nothing))
    @test_nowarn show(stdout, indicator_loehner)

    indicator_max = IndicatorMax("variable", (; cache=nothing))
    @test_nowarn show(stdout, indicator_max)

    equations = CompressibleEulerEquations2D(1.4)
    basis = LobattoLegendreBasis(3)
    indicator_neuralnetwork = IndicatorNeuralNetwork(
        equations, basis, indicator_type=NeuralNetworkPerssonPeraire(), variable=density,
        network=nothing)
    @test_nowarn show(stdout, indicator_neuralnetwork)
  end

  @timed_testset "LBM 2D constructor" begin
    # Neither Mach number nor velocity set
    @test_throws ErrorException LatticeBoltzmannEquations2D(Ma=nothing, Re=1000)
    # Both Mach number and velocity set
    @test_throws ErrorException LatticeBoltzmannEquations2D(Ma=0.1, Re=1000, u0=1)
    # Neither Reynolds number nor viscosity set
    @test_throws ErrorException LatticeBoltzmannEquations2D(Ma=0.1, Re=nothing)
    # Both Reynolds number and viscosity set
    @test_throws ErrorException LatticeBoltzmannEquations2D(Ma=0.1, Re=1000, nu=1)

    # No non-dimensional values set
    @test LatticeBoltzmannEquations2D(Ma=nothing, Re=nothing, u0=1, nu=1) isa LatticeBoltzmannEquations2D
  end

  @timed_testset "LBM 3D constructor" begin
    # Neither Mach number nor velocity set
    @test_throws ErrorException LatticeBoltzmannEquations3D(Ma=nothing, Re=1000)
    # Both Mach number and velocity set
    @test_throws ErrorException LatticeBoltzmannEquations3D(Ma=0.1, Re=1000, u0=1)
    # Neither Reynolds number nor viscosity set
    @test_throws ErrorException LatticeBoltzmannEquations3D(Ma=0.1, Re=nothing)
    # Both Reynolds number and viscosity set
    @test_throws ErrorException LatticeBoltzmannEquations3D(Ma=0.1, Re=1000, nu=1)

    # No non-dimensional values set
    @test LatticeBoltzmannEquations3D(Ma=nothing, Re=nothing, u0=1, nu=1) isa LatticeBoltzmannEquations3D
  end

  @timed_testset "LBM 2D functions" begin
    # Set up LBM struct and dummy distribution
    equation = LatticeBoltzmannEquations2D(Ma=0.1, Re=1000)
    u = Trixi.equilibrium_distribution(1, 2, 3, equation)

    # Component-wise velocity
    @test isapprox(Trixi.velocity(u, 1, equation), 2)
    @test isapprox(Trixi.velocity(u, 2, equation), 3)
  end

  @timed_testset "LBM 3D functions" begin
    # Set up LBM struct and dummy distribution
    equation = LatticeBoltzmannEquations3D(Ma=0.1, Re=1000)
    u = Trixi.equilibrium_distribution(1, 2, 3, 4, equation)

    # Component-wise velocity
    @test isapprox(velocity(u, 1, equation), 2)
    @test isapprox(velocity(u, 2, equation), 3)
    @test isapprox(velocity(u, 3, equation), 4)
  end

  @timed_testset "LBMCollisionCallback" begin
    # Printing of LBM collision callback
    callback = LBMCollisionCallback()
    @test_nowarn show(stdout, callback)
    println()
    @test_nowarn show(stdout, "text/plain", callback)
    println()
  end

  @timed_testset "Acoustic perturbation 2D varnames" begin
    v_mean_global = (0.0, 0.0)
    c_mean_global = 1.0
    rho_mean_global = 1.0
    equations = AcousticPerturbationEquations2D(v_mean_global, c_mean_global, rho_mean_global)

    @test Trixi.varnames(cons2state, equations) == ("v1_prime", "v2_prime", "p_prime_scaled")
    @test Trixi.varnames(cons2mean, equations) == ("v1_mean", "v2_mean", "c_mean", "rho_mean")
  end

  @timed_testset "Euler conversion between conservative/entropy variables" begin
    rho, v1, v2, v3, p = 1.0, 0.1, 0.2, 0.3, 2.0

    let equations = CompressibleEulerEquations1D(1.4)
      cons_vars = prim2cons(SVector(rho, v1, p),equations)
      entropy_vars = cons2entropy(cons_vars, equations)
      @test cons_vars ≈ entropy2cons(entropy_vars, equations)

      # test tuple args
      cons_vars = prim2cons((rho, v1, p),equations)
      entropy_vars = cons2entropy(cons_vars, equations)
      @test cons_vars ≈ entropy2cons(entropy_vars, equations)
    end

    let equations = CompressibleEulerEquations2D(1.4)
      cons_vars = prim2cons(SVector(rho,v1,v2,p),equations)
      entropy_vars = cons2entropy(cons_vars,equations)
      @test cons_vars ≈ entropy2cons(entropy_vars,equations)

      # test tuple args
      cons_vars = prim2cons((rho, v1, v2, p), equations)
      entropy_vars = cons2entropy(cons_vars, equations)
      @test cons_vars ≈ entropy2cons(entropy_vars, equations)
    end

    let equations = CompressibleEulerEquations3D(1.4)
      cons_vars = prim2cons(SVector(rho,v1,v2,v3,p),equations)
      entropy_vars = cons2entropy(cons_vars,equations)
      @test cons_vars ≈ entropy2cons(entropy_vars,equations)

      # test tuple args
      cons_vars = prim2cons((rho, v1, v2, v3, p), equations)
      entropy_vars = cons2entropy(cons_vars, equations)
      @test cons_vars ≈ entropy2cons(entropy_vars, equations)
    end
  end

  @timed_testset "Shallow water conversion between conservative/entropy variables" begin
    H, v1, v2, b = 3.5, 0.25, 0.1, 0.4

    let equations = ShallowWaterEquations1D(gravity_constant=9.8)
      cons_vars = prim2cons(SVector(H, v1, b),equations)
      entropy_vars = cons2entropy(cons_vars,equations)
      @test cons_vars ≈ entropy2cons(entropy_vars,equations)

      total_energy = energy_total(cons_vars, equations)
      @test total_energy ≈ entropy(cons_vars, equations)

      # test tuple args
      cons_vars = prim2cons((H, v1, b), equations)
      entropy_vars = cons2entropy(cons_vars, equations)
      @test cons_vars ≈ entropy2cons(entropy_vars, equations)
    end

    let equations = ShallowWaterEquations2D(gravity_constant=9.8)
      cons_vars = prim2cons(SVector(H,v1,v2,b),equations)
      entropy_vars = cons2entropy(cons_vars,equations)
      @test cons_vars ≈ entropy2cons(entropy_vars,equations)

      total_energy = energy_total(cons_vars, equations)
      @test total_energy ≈ entropy(cons_vars, equations)

      # test tuple args
      cons_vars = prim2cons((H, v1, v2, b), equations)
      entropy_vars = cons2entropy(cons_vars, equations)
      @test cons_vars ≈ entropy2cons(entropy_vars, equations)
    end
  end

  @timed_testset "boundary_condition_do_nothing" begin
    rho, v1, v2, p = 1.0, 0.1, 0.2, 0.3, 2.0

    let equations = CompressibleEulerEquations2D(1.4)
      u = prim2cons(SVector(rho, v1, v2, p), equations)
      x = SVector(1.0, 2.0)
      t = 0.5
      surface_flux = flux_lax_friedrichs

      outward_direction = SVector(0.2, -0.3)
      @test flux(u, outward_direction, equations) ≈ boundary_condition_do_nothing(u, outward_direction, x, t, surface_flux, equations)

      orientation = 2
      direction = 4
      @test flux(u, orientation, equations) ≈ boundary_condition_do_nothing(u, orientation, direction, x, t, surface_flux, equations)
    end
  end

  @timed_testset "TimeSeriesCallback" begin
    @test_nowarn_mod trixi_include(@__MODULE__,
                                     joinpath(examples_dir(), "tree_2d_dgsem", "elixir_acoustics_gaussian_source.jl"),
                                     tspan=(0, 0.05))

    point_data_1 = time_series.affect!.point_data[1]
    @test all(isapprox.(point_data_1[1:7], [-2.4417734981719132e-5, -3.4296207289200194e-5,
                                            0.0018130846385739788, -0.5, 0.25, 1.0, 1.0]))
    @test_throws DimensionMismatch Trixi.get_elements_by_coordinates!([1, 2], rand(2, 4), mesh,
                                                                      solver, nothing)
    @test_nowarn show(stdout, time_series)
    @test_throws ArgumentError TimeSeriesCallback(semi, [(1.0, 1.0)]; interval=-1)
    @test_throws ArgumentError TimeSeriesCallback(semi, [1.0 1.0 1.0; 2.0 2.0 2.0])
  end

  @timed_testset "Consistency check for HLLE flux" begin
    # Set up equations and dummy conservative variables state
    equations = CompressibleEulerEquations1D(1.4)
    u = SVector(1.1, 2.34, 5.5)

    orientations = [1]
    for orientation in orientations
      @test flux_hlle(u, u, orientation, equations) ≈ flux(u, orientation, equations)
    end

    equations = CompressibleEulerEquations2D(1.4)
    u = SVector(1.1, -0.5, 2.34, 5.5)

    orientations = [1, 2]
    for orientation in orientations
      @test flux_hlle(u, u, orientation, equations) ≈ flux(u, orientation, equations)
    end

    equations = CompressibleEulerEquations3D(1.4)
    u = SVector(1.1, -0.5, 2.34, 2.4, 5.5)

    orientations = [1, 2, 3]
    for orientation in orientations
      @test flux_hlle(u, u, orientation, equations) ≈ flux(u, orientation, equations)
    end
  end

  @timed_testset "Consistency check for Godunov flux" begin
    # Set up equations and dummy conservative variables state
    # Burgers' Equation

    equation = InviscidBurgersEquation1D()
    u_values = [SVector(42.0), SVector(-42.0)]

    orientations = [1]
    for orientation in orientations, u in u_values
      @test flux_godunov(u, u, orientation, equation) ≈ flux(u, orientation, equation)
    end

    # Linear Advection 1D
    equation = LinearScalarAdvectionEquation1D(-4.2)
    u = SVector(3.14159)

    orientations = [1]
    for orientation in orientations
      @test flux_godunov(u, u, orientation, equation) ≈ flux(u, orientation, equation)
    end

    # Linear Advection 2D
    equation = LinearScalarAdvectionEquation2D(-4.2, 2.4)
    u = SVector(3.14159)

    orientations = [1, 2]
    for orientation in orientations
      @test flux_godunov(u, u, orientation, equation) ≈ flux(u, orientation, equation)
    end

    normal_directions = [SVector(1.0, 0.0),
                         SVector(0.0, 1.0),
                         SVector(0.5, -0.5),
                         SVector(-1.2, 0.3)]

    for normal_direction in normal_directions
      @test flux_godunov(u, u, normal_direction, equation) ≈ flux(u, normal_direction, equation)
    end

    # Linear Advection 3D
    equation = LinearScalarAdvectionEquation3D(-4.2, 2.4, 1.2)
    u = SVector(3.14159)

    orientations = [1, 2, 3]
    for orientation in orientations
      @test flux_godunov(u, u, orientation, equation) ≈ flux(u, orientation, equation)
    end

    normal_directions = [SVector(1.0, 0.0, 0.0),
                         SVector(0.0, 1.0, 0.0),
                         SVector(0.0, 0.0, 1.0),
                         SVector(0.5, -0.5, 0.2),
                         SVector(-1.2, 0.3, 1.4)]

    for normal_direction in normal_directions
      @test flux_godunov(u, u, normal_direction, equation) ≈ flux(u, normal_direction, equation)
    end
  end

  @timed_testset "Consistency check for Engquist-Osher flux" begin
    # Set up equations and dummy conservative variables state
    equation = InviscidBurgersEquation1D()
    u_values = [SVector(42.0), SVector(-42.0)]

    orientations = [1]
    for orientation in orientations, u in u_values
      @test Trixi.flux_engquist_osher(u, u, orientation, equation) ≈ flux(u, orientation, equation)
    end

    equation = LinearScalarAdvectionEquation1D(-4.2)
    u = SVector(3.14159)

    orientations = [1]
    for orientation in orientations
      @test Trixi.flux_engquist_osher(u, u, orientation, equation) ≈ flux(u, orientation, equation)
    end
  end

  @testset "Equivalent Fluxes" begin
    # Set up equations and dummy conservative variables state
    # Burgers' Equation

    equation = InviscidBurgersEquation1D()
    u_values = [SVector(42.0), SVector(-42.0)]

    orientations = [1]
    for orientation in orientations, u in u_values
      @test flux_godunov(0.75*u, u, orientation, equation) ≈ Trixi.flux_engquist_osher(0.75*u, u, orientation, equation)
    end

    # Linear Advection 1D
    equation = LinearScalarAdvectionEquation1D(-4.2)
    u = SVector(3.14159)

    orientations = [1]
    for orientation in orientations
      @test flux_godunov(0.5*u, u, orientation, equation) ≈ flux_lax_friedrichs(0.5*u, u, orientation, equation)
      @test flux_godunov(2*u, u, orientation, equation) ≈ Trixi.flux_engquist_osher(2*u, u, orientation, equation)
    end

    # Linear Advection 2D
    equation = LinearScalarAdvectionEquation2D(-4.2, 2.4)
    u = SVector(3.14159)

    orientations = [1, 2]
    for orientation in orientations
      @test flux_godunov(0.25*u, u, orientation, equation) ≈ flux_lax_friedrichs(0.25*u, u, orientation, equation)
    end

    normal_directions = [SVector(1.0, 0.0),
                         SVector(0.0, 1.0),
                         SVector(0.5, -0.5),
                         SVector(-1.2, 0.3)]

    for normal_direction in normal_directions
      @test flux_godunov(3*u, u, normal_direction, equation) ≈ flux_lax_friedrichs(3*u, u, normal_direction, equation)
    end

    # Linear Advection 3D
    equation = LinearScalarAdvectionEquation3D(-4.2, 2.4, 1.2)
    u = SVector(3.14159)

    orientations = [1, 2, 3]
    for orientation in orientations
      @test flux_godunov(1.5*u, u, orientation, equation) ≈ flux_lax_friedrichs(1.5*u, u, orientation, equation)
    end

    normal_directions = [SVector(1.0, 0.0, 0.0),
                         SVector(0.0, 1.0, 0.0),
                         SVector(0.0, 0.0, 1.0),
                         SVector(0.5, -0.5, 0.2),
                         SVector(-1.2, 0.3, 1.4)]

    for normal_direction in normal_directions
      @test flux_godunov(1.3*u, u, normal_direction, equation) ≈ flux_lax_friedrichs(1.3*u, u, normal_direction, equation)
    end
  end

  @testset "FluxRotated vs. direct implementation" begin
    @timed_testset "CompressibleEulerEquations2D" begin
      equations = CompressibleEulerEquations2D(1.4)
      normal_directions = [SVector(1.0, 0.0),
                           SVector(0.0, 1.0),
                           SVector(0.5, -0.5),
                           SVector(-1.2, 0.3)]
      u_values = [SVector(1.0, 0.5, -0.7, 1.0),
                  SVector(1.5, -0.2, 0.1, 5.0),]
      fluxes = [flux_central, flux_ranocha, flux_shima_etal, flux_kennedy_gruber]

      for f_std in fluxes
        f_rot = FluxRotated(f_std)
        for u_ll in u_values, u_rr in u_values, normal_direction in normal_directions
          @test f_rot(u_ll, u_rr, normal_direction, equations) ≈ f_std(u_ll, u_rr, normal_direction, equations)
        end
      end
    end

    @timed_testset "CompressibleEulerEquations3D" begin
      equations = CompressibleEulerEquations3D(1.4)
      normal_directions = [SVector(1.0, 0.0, 0.0),
                          SVector(0.0, 1.0, 0.0),
                          SVector(0.0, 0.0, 1.0),
                          SVector(0.5, -0.5, 0.2),
                          SVector(-1.2, 0.3, 1.4)]
      u_values = [SVector(1.0, 0.5, -0.7, 0.1, 1.0),
                  SVector(1.5, -0.2, 0.1, 0.2, 5.0),]
      fluxes = [flux_central, flux_ranocha, flux_shima_etal, flux_kennedy_gruber, FluxLMARS(340)]

      for f_std in fluxes
        f_rot = FluxRotated(f_std)
        for u_ll in u_values, u_rr in u_values, normal_direction in normal_directions
          @test f_rot(u_ll, u_rr, normal_direction, equations) ≈ f_std(u_ll, u_rr, normal_direction, equations)
        end
      end
    end

    @timed_testset "IdealGlmMhdEquations2D" begin
      equations = IdealGlmMhdEquations2D(1.4, 5.0 #= c_h =#)
      normal_directions = [SVector(1.0, 0.0),
                           SVector(0.0, 1.0),
                           SVector(0.5, -0.5),
                           SVector(-1.2, 0.3)]
      u_values = [SVector(1.0, 0.4, -0.5, 0.1, 1.0, 0.1, -0.2, 0.1, 0.0),
                  SVector(1.5, -0.2, 0.1, 0.2, 5.0, -0.1, 0.1, 0.2, 0.2),]
      fluxes = [flux_central, flux_hindenlang_gassner]

      for f_std in fluxes
        f_rot = FluxRotated(f_std)
        for u_ll in u_values, u_rr in u_values, normal_direction in normal_directions
          @test f_rot(u_ll, u_rr, normal_direction, equations) ≈ f_std(u_ll, u_rr, normal_direction, equations)
        end
      end
    end

    @timed_testset "IdealGlmMhdEquations3D" begin
      equations = IdealGlmMhdEquations3D(1.4, 5.0 #= c_h =#)
      normal_directions = [SVector(1.0, 0.0, 0.0),
                          SVector(0.0, 1.0, 0.0),
                          SVector(0.0, 0.0, 1.0),
                          SVector(0.5, -0.5, 0.2),
                          SVector(-1.2, 0.3, 1.4)]
      u_values = [SVector(1.0, 0.4, -0.5, 0.1, 1.0, 0.1, -0.2, 0.1, 0.0),
                  SVector(1.5, -0.2, 0.1, 0.2, 5.0, -0.1, 0.1, 0.2, 0.2),]
      fluxes = [flux_central, flux_hindenlang_gassner]

      for f_std in fluxes
        f_rot = FluxRotated(f_std)
        for u_ll in u_values, u_rr in u_values, normal_direction in normal_directions
          @test f_rot(u_ll, u_rr, normal_direction, equations) ≈ f_std(u_ll, u_rr, normal_direction, equations)
        end
      end
    end
  end

  @testset "SimpleKronecker" begin
    N = 3

    NDIMS = 2
    r, s = StartUpDG.nodes(Quad(), N)
    V = StartUpDG.vandermonde(Quad(), N, r, s)
    r1D = StartUpDG.nodes(Line(), N)
    V1D = StartUpDG.vandermonde(Line(), N, r1D)

    x = r + s
    V_kron = Trixi.SimpleKronecker(NDIMS, V1D, eltype(x))

    b = similar(x)
    b_kron = similar(x)
    Trixi.mul!(b, V, x)
    Trixi.mul!(b_kron, V_kron, x)
    @test b ≈ b_kron
  end

  @testset "SummationByPartsOperators + StartUpDG" begin
    dg = DGMulti(polydeg = 3, element_type = Quad(),
                 approximation_type = derivative_operator(
                 SummationByPartsOperators.MattssonNordström2004(),
                 derivative_order=1, accuracy_order=4,
                 xmin=0.0, xmax=1.0, N=10))

    @test StartUpDG.inverse_trace_constant(dg.basis) ≈ 50.8235294117647
  end

  @testset "1D non-periodic DGMultiMesh" begin
    # checks whether or not boundary faces are initialized correctly for DGMultiMesh in 1D
    dg = DGMulti(polydeg = 1, element_type = Line(), approximation_type = Polynomial(),
                 surface_integral = SurfaceIntegralWeakForm(flux_central),
                 volume_integral = VolumeIntegralFluxDifferencing(flux_central))
    mesh = DGMultiMesh(dg, cells_per_dimension=(1,), periodicity=false)

    @test mesh.boundary_faces[:entire_boundary] == [1, 2]
  end

end



end #module
