using Test
using MPI: mpiexec

# We run tests in parallel with CI jobs setting the `TRIXI_TEST` environment
# variable to determine the subset of tests to execute.
# By default, we just run the threaded tests since they are relatively cheap
# and test a good amount of different functionality.
const TRIXI_TEST = get(ENV, "TRIXI_TEST", "threaded")
const TRIXI_MPI_NPROCS = clamp(Sys.CPU_THREADS, 2, 3)
const TRIXI_NTHREADS = clamp(Sys.CPU_THREADS, 2, 3)

@time @testset "Trixi.jl tests" begin
    # This is placed first since tests error out otherwise if `TRIXI_TEST == "all"`,
    # at least on some systems.
    @time if TRIXI_TEST == "all" || TRIXI_TEST == "mpi"
        # Do a dummy `@test true`:
        # If the process errors out the testset would error out as well,
        # cf. https://github.com/JuliaParallel/MPI.jl/pull/391
        @test true

        # We provide a `--heap-size-hint` to avoid/reduce out-of-memory errors during CI testing
        mpiexec() do cmd
            run(`$cmd -n $TRIXI_MPI_NPROCS $(Base.julia_cmd()) --threads=1 --check-bounds=yes --heap-size-hint=0.5G $(abspath("test_mpi.jl"))`)
        end
    end

    @time if TRIXI_TEST == "all" || TRIXI_TEST == "threaded" ||
             TRIXI_TEST == "threaded_legacy"
        # Do a dummy `@test true`:
        # If the process errors out the testset would error out as well,
        # cf. https://github.com/JuliaParallel/MPI.jl/pull/391
        @test true

        run(`$(Base.julia_cmd()) --threads=$TRIXI_NTHREADS --check-bounds=yes --code-coverage=none $(abspath("test_threaded.jl"))`)
    end

    @time if TRIXI_TEST == "all" || TRIXI_TEST == "tree_part1"
        include("test_tree_1d.jl")
        include("test_tree_2d_part1.jl")
    end
    @time if TRIXI_TEST == "all" || TRIXI_TEST == "tree_part2"
        include("test_tree_2d_part2.jl")
    end
    @time if TRIXI_TEST == "all" || TRIXI_TEST == "tree_part3"
        include("test_tree_2d_part3.jl")
    end
    @time if TRIXI_TEST == "all" || TRIXI_TEST == "tree_part4"
        include("test_tree_3d_part1.jl")
    end
    @time if TRIXI_TEST == "all" || TRIXI_TEST == "tree_part5"
        include("test_tree_3d_part2.jl")
    end
    @time if TRIXI_TEST == "all" || TRIXI_TEST == "tree_part6"
        include("test_tree_3d_part3.jl")
    end

    @time if TRIXI_TEST == "all" || TRIXI_TEST == "structured_part1"
        include("test_structured_1d.jl")
        include("test_structured_3d.jl")
    end
    @time if TRIXI_TEST == "all" || TRIXI_TEST == "structured_part2"
        include("test_structured_2d.jl")
    end

    @time if TRIXI_TEST == "all" || TRIXI_TEST == "p4est_part1"
        include("test_p4est_2d.jl")
    end
    @time if TRIXI_TEST == "all" || TRIXI_TEST == "p4est_part2"
        include("test_p4est_3d.jl")
    end

    @time if TRIXI_TEST == "all" || TRIXI_TEST == "t8code_part1"
        include("test_t8code_2d.jl")
    end
    @time if TRIXI_TEST == "all" || TRIXI_TEST == "t8code_part2"
        include("test_t8code_3d.jl")
    end

    @time if TRIXI_TEST == "all" || TRIXI_TEST == "unstructured"
        include("test_unstructured_2d.jl")
    end

    @time if TRIXI_TEST == "all" || TRIXI_TEST == "dgmulti_part1"
        include("test_dgmulti_1d.jl")
        include("test_dgmulti_3d.jl")
    end
    @time if TRIXI_TEST == "all" || TRIXI_TEST == "dgmulti_part2"
        include("test_dgmulti_2d.jl")
    end

    @time if TRIXI_TEST == "all" || TRIXI_TEST == "parabolic_part1"
        include("test_parabolic_1d.jl")
    end
    @time if TRIXI_TEST == "all" || TRIXI_TEST == "parabolic_part2"
        include("test_parabolic_2d.jl")
    end
    @time if TRIXI_TEST == "all" || TRIXI_TEST == "parabolic_part3"
        include("test_parabolic_3d.jl")
    end

    @time if TRIXI_TEST == "all" || TRIXI_TEST == "misc_part1"
        include("test_unit.jl")
        include("test_type.jl")
        include("test_visualization.jl")
    end
    @time if TRIXI_TEST == "all" || TRIXI_TEST == "misc_part2"
        include("test_special_elixirs.jl")
        include("test_aqua.jl")
    end

    @time if TRIXI_TEST == "all" || TRIXI_TEST == "performance_specializations_part1"
        include("test_performance_specializations_2d.jl")
    end
    @time if TRIXI_TEST == "all" || TRIXI_TEST == "performance_specializations_part2"
        include("test_performance_specializations_3d.jl")
    end

    @time if TRIXI_TEST == "all" || TRIXI_TEST == "paper_self_gravitating_gas_dynamics"
        include("test_paper_self_gravitating_gas_dynamics.jl")
    end
end
