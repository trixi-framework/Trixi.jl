using Test
using MPI: mpiexec

# run tests on Travis CI in parallel
const TRIXI_TEST = get(ENV, "TRIXI_TEST", "all")
const ON_APPVEYOR = lowercase(get(ENV, "APPVEYOR", "false")) == "true"
const TRIXI_MPI_NPROCS = 3

@time @testset "Trixi.jl tests" begin
  @time if TRIXI_TEST == "all" || TRIXI_TEST == "1D"
    include("test_examples_1d.jl")
  end

  @time if TRIXI_TEST == "all" || TRIXI_TEST == "2D"
    include("test_examples_2d.jl")
  end

  @time if (TRIXI_TEST == "all" && !ON_APPVEYOR) || TRIXI_TEST == "3D"
    include("test_examples_3d.jl")
  end

  @time if TRIXI_TEST == "all" || TRIXI_TEST == "misc"
    include("test_manual.jl")
    include("test_elixirs.jl")
  end

  @time if (TRIXI_TEST == "all" && !ON_APPVEYOR) || TRIXI_TEST == "paper-self-gravitating-gas-dynamics"
    include("test_paper-self-gravitating-gas-dynamics.jl")
  end

  @time if TRIXI_TEST == "all" || TRIXI_TEST == "parallel_2d"
    mpiexec() do cmd
      run(`$cmd -n $TRIXI_MPI_NPROCS $(Base.julia_cmd()) test_examples_parallel_2d.jl`)
    end
  end
end
