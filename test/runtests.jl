using Test
using MPI: mpiexec

# run tests on Travis CI in parallel
const TRIXI_TEST = get(ENV, "TRIXI_TEST", "all")
const ON_APPVEYOR = lowercase(get(ENV, "APPVEYOR", "false")) == "true"
const TRIXI_MPI_NPROCS = clamp(Sys.CPU_THREADS, 2, 3)

@time @testset "Trixi.jl tests" begin
  @time if TRIXI_TEST == "all" || TRIXI_TEST == "1D"
    include("test_examples_1d.jl")
    include("test_examples_1d_old.jl") # TODO: Taal remove
  end

  @time if TRIXI_TEST == "all" || TRIXI_TEST == "2D"
    include("test_examples_2d.jl")
  end

  @time if TRIXI_TEST == "all" || TRIXI_TEST == "2D_OLD" # TODO: Taal remove
    include("test_examples_2d_old.jl")
  end

  @time if (TRIXI_TEST == "all" && !ON_APPVEYOR) || TRIXI_TEST == "3D"
    include("test_examples_3d.jl")
  end

  @time if (TRIXI_TEST == "all" && !ON_APPVEYOR) || TRIXI_TEST == "3D_OLD" # TODO: Taal remove
    include("test_examples_3d_old.jl")
  end

  @time if TRIXI_TEST == "all" || TRIXI_TEST == "misc"
    include("test_manual.jl")
    include("test_special_elixirs.jl")
    include("test_special_elixirs_old.jl") # TODO: Taal remove
  end

  @time if (TRIXI_TEST == "all" && !ON_APPVEYOR) || TRIXI_TEST == "paper-self-gravitating-gas-dynamics"
    include("test_paper-self-gravitating-gas-dynamics.jl")
  end

  @time if TRIXI_TEST == "all" || TRIXI_TEST == "parallel_2d"
    # Based on `runtests.jl` from `MPI.jl` and `PencilArrays.jl`
    # Precompilation disabled to prevent race conditions when loading packages
    mpiexec() do cmd
      run(`$cmd -n $TRIXI_MPI_NPROCS $(Base.julia_cmd()) --compiled-modules=no --threads=1 --check-bounds=yes test_examples_parallel_2d.jl`)
    end
  end
end
