using Test
using MPI: mpiexec

# run tests on Travis CI in parallel
const TRIXI_TEST = get(ENV, "TRIXI_TEST", "all")
const TRIXI_MPI_NPROCS = clamp(Sys.CPU_THREADS, 2, 3)
const TRIXI_NTHREADS   = clamp(Sys.CPU_THREADS, 2, 3)

@time @testset "Trixi.jl tests" begin
  # This is placed first since tests error out otherwise if `TRIXI_TEST == "all"`,
  # at least on some systems.
  @time if TRIXI_TEST == "all" || TRIXI_TEST == "2d_mpi"
    # Do a dummy `@test true`:
    # If the process errors out the testset would error out as well,
    # cf. https://github.com/JuliaParallel/MPI.jl/pull/391
    @test true

    # Based on `runtests.jl` from `MPI.jl` and `PencilArrays.jl`
    # On Julia v1.5 and before, precompilation is strictly serial and any attempt
    # to use it in parallel will result in race conditions and probably errors.
    # Hence, the additional flag `--compiled-modules=no` is required for Julia
    # versions older than v1.6.
    mpiexec() do cmd
      run(`$cmd -n $TRIXI_MPI_NPROCS $(Base.julia_cmd()) --threads=1 --check-bounds=yes test_examples_2d_parallel.jl`)
    end
  end

  @time if TRIXI_TEST == "all" || TRIXI_TEST == "2d_threaded"
    # Do a dummy `@test true`:
    # If the process errors out the testset would error out as well,
    # cf. https://github.com/JuliaParallel/MPI.jl/pull/391
    @test true

    run(`$(Base.julia_cmd()) --threads=$TRIXI_NTHREADS --check-bounds=yes --code-coverage=none test_examples_2d_parallel.jl`)
  end

  @time if TRIXI_TEST == "all" || TRIXI_TEST == "1d"
    include("test_examples_1d.jl")
  end

  @time if TRIXI_TEST == "all" || TRIXI_TEST == "2d_part1"
    include("test_examples_2d_part1.jl")
  end

  @time if TRIXI_TEST == "all" || TRIXI_TEST == "2d_part2"
    include("test_examples_2d_part2.jl")
  end

  @time if TRIXI_TEST == "all" || TRIXI_TEST == "2d_part3"
    include("test_examples_2d_part3.jl")
  end

  @time if TRIXI_TEST == "all" || TRIXI_TEST == "3d_part1"
    include("test_examples_3d_part1.jl")
  end

  @time if TRIXI_TEST == "all" || TRIXI_TEST == "3d_part2"
    include("test_examples_3d_part2.jl")
  end

  @time if TRIXI_TEST == "all" || TRIXI_TEST == "misc"
    include("test_unit.jl")
    include("test_special_elixirs.jl")
    include("test_visualization.jl")
  end

  @time if TRIXI_TEST == "all" || TRIXI_TEST == "paper-self-gravitating-gas-dynamics"
    include("test_paper-self-gravitating-gas-dynamics.jl")
  end
end
