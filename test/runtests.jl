using Test
using MPI: mpiexec

# run tests on Travis CI in parallel
const TRIXI_TEST = get(ENV, "TRIXI_TEST", "all")
const TRIXI_MPI_NPROCS = clamp(Sys.CPU_THREADS, 2, 3)

@time @testset "Trixi.jl tests" begin
  @time if TRIXI_TEST == "all" || TRIXI_TEST == "1d"
    include("test_examples_1d.jl")
  end

  @time if TRIXI_TEST == "all" || TRIXI_TEST == "2d"
    include("test_examples_2d.jl")
  end

  @time if TRIXI_TEST == "all" || TRIXI_TEST == "2d_parallel"
    # Do a dummy `@test true`:
    # If the process errors out the testset would error out as well,
    # cf. https://github.com/JuliaParallel/MPI.jl/pull/391
    @test true

    # Based on `runtests.jl` from `MPI.jl` and `PencilArrays.jl`
    # Precompilation disabled to prevent race conditions when loading packages
    # TODO: We can remove the flag `--compiled-modules=no` on Julia v1.6.
    mpiexec() do cmd
      run(`$cmd -n $TRIXI_MPI_NPROCS $(Base.julia_cmd()) --compiled-modules=no --threads=1 --check-bounds=yes test_examples_2d_parallel.jl`)
    end
  end

  @time if TRIXI_TEST == "all" || TRIXI_TEST == "3d"
    include("test_examples_3d.jl")
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
