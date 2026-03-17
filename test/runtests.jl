using Test
using MPI: mpiexec
import Trixi

# We run tests in parallel with CI jobs setting the `TRIXI_TEST` environment
# variable to determine the subset of tests to execute.
# By default, we just run the threaded tests since they are relatively cheap
# and test a good amount of different functionality.
const TRIXI_TEST = get(ENV, "TRIXI_TEST", "threaded")
# Some GitHub CI runners may have not much RAM and just 3 virtual CPU cores.
# In this case, we do not want to use all of the cores to speed-up CI.
const TRIXI_MPI_NPROCS = clamp(Sys.CPU_THREADS - 1, 2, 3)
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
            run(`$cmd -n $TRIXI_MPI_NPROCS $(Base.julia_cmd()) --threads=1 --check-bounds=yes --heap-size-hint=0.5G $(joinpath(@__DIR__, "test_mpi.jl"))`)
            return nothing
        end
    end

    @time if TRIXI_TEST == "all" || TRIXI_TEST == "threaded" ||
             TRIXI_TEST == "threaded_legacy"
        # Do a dummy `@test true`:
        # If the process errors out the testset would error out as well,
        # cf. https://github.com/JuliaParallel/MPI.jl/pull/391
        @test true

        run(`$(Base.julia_cmd()) --threads=$TRIXI_NTHREADS --check-bounds=yes --code-coverage=none $(joinpath(@__DIR__, "test_threaded.jl"))`)
    end

    # Downgrade CI currently has issues with running julia processes via `run`, see
    # https://github.com/trixi-framework/Trixi.jl/pull/2507#issuecomment-3990318366
    # So we run test_threaded.jl serially.
    # For `TRIXI_TEST = "all"`, test_threaded.jl is already covered by the threaded run, so we don't need to run it again.
    @time if TRIXI_TEST == "downgrade"
        include(joinpath(@__DIR__, "test_threaded.jl"))
    end

    @time if TRIXI_TEST == "all" || TRIXI_TEST == "tree_part1"
        include(joinpath(@__DIR__, "test_tree_1d.jl"))
        include(joinpath(@__DIR__, "test_tree_2d_part1.jl"))
    end
    @time if TRIXI_TEST == "all" || TRIXI_TEST == "tree_part2"
        include(joinpath(@__DIR__, "test_tree_2d_part2.jl"))
    end
    @time if TRIXI_TEST == "all" || TRIXI_TEST == "tree_part3"
        include(joinpath(@__DIR__, "test_tree_2d_part3.jl"))
    end
    @time if TRIXI_TEST == "all" || TRIXI_TEST == "tree_part4"
        include(joinpath(@__DIR__, "test_tree_3d_part1.jl"))
    end
    @time if TRIXI_TEST == "all" || TRIXI_TEST == "tree_part5"
        include(joinpath(@__DIR__, "test_tree_3d_part2.jl"))
    end
    @time if TRIXI_TEST == "all" || TRIXI_TEST == "tree_part6"
        include(joinpath(@__DIR__, "test_tree_3d_part3.jl"))
    end

    @time if TRIXI_TEST == "all" || TRIXI_TEST == "structured"
        include(joinpath(@__DIR__, "test_structured_1d.jl"))
        include(joinpath(@__DIR__, "test_structured_2d.jl"))
        include(joinpath(@__DIR__, "test_structured_3d.jl"))
    end

    @time if TRIXI_TEST == "all" || TRIXI_TEST == "p4est_part1"
        include(joinpath(@__DIR__, "test_p4est_2d.jl"))
    end
    @time if TRIXI_TEST == "all" || TRIXI_TEST == "p4est_part2"
        include(joinpath(@__DIR__, "test_p4est_3d.jl"))
    end

    @time if TRIXI_TEST == "all" || TRIXI_TEST == "t8code_part1"
        include(joinpath(@__DIR__, "test_t8code_2d.jl"))
    end
    @time if TRIXI_TEST == "all" || TRIXI_TEST == "t8code_part2"
        include(joinpath(@__DIR__, "test_t8code_3d.jl"))
    end

    @time if TRIXI_TEST == "all" || TRIXI_TEST == "unstructured_dgmulti"
        include(joinpath(@__DIR__, "test_unstructured_2d.jl"))
        include(joinpath(@__DIR__, "test_dgmulti_1d.jl"))
        include(joinpath(@__DIR__, "test_dgmulti_2d.jl"))
        include(joinpath(@__DIR__, "test_dgmulti_3d.jl"))
    end

    @time if TRIXI_TEST == "all" || TRIXI_TEST == "parabolic_part1"
        include(joinpath(@__DIR__, "test_parabolic_1d.jl"))
        include(joinpath(@__DIR__, "test_parabolic_2d.jl"))
    end
    @time if TRIXI_TEST == "all" || TRIXI_TEST == "parabolic_part2"
        include(joinpath(@__DIR__, "test_parabolic_3d.jl"))
    end

    @time if TRIXI_TEST == "all" || TRIXI_TEST == "misc_part1"
        include(joinpath(@__DIR__, "test_unit.jl"))
        include(joinpath(@__DIR__, "test_type.jl"))
        include(joinpath(@__DIR__, "test_visualization.jl"))
    end
    @time if TRIXI_TEST == "all" || TRIXI_TEST == "misc_part2"
        include(joinpath(@__DIR__, "test_special_elixirs.jl"))
        include(joinpath(@__DIR__, "test_aqua.jl"))
    end

    @time if TRIXI_TEST == "all" || TRIXI_TEST == "performance_specializations"
        include(joinpath(@__DIR__, "test_performance_specializations_2d.jl"))
        include(joinpath(@__DIR__, "test_performance_specializations_3d.jl"))
    end

    @time if TRIXI_TEST == "all" || TRIXI_TEST == "paper_self_gravitating_gas_dynamics"
        include(joinpath(@__DIR__, "test_paper_self_gravitating_gas_dynamics.jl"))
    end

    @time if TRIXI_TEST == "all" || TRIXI_TEST == "CUDA"
        import CUDA
        if CUDA.functional()
            include(joinpath(@__DIR__, "test_cuda_2d.jl"))
            include(joinpath(@__DIR__, "test_cuda_3d.jl"))
        else
            @warn "Unable to run CUDA tests on this machine"
        end
    end

    @time if TRIXI_TEST == "all" || TRIXI_TEST == "AMDGPU"
        import AMDGPU
        if AMDGPU.functional()
            include("test_amdgpu_2d.jl")
        else
            @warn "Unable to run AMDGPU tests on this machine"
        end
    end

    @time if TRIXI_TEST == "all" || TRIXI_TEST == "kernelabstractions"
        previous_backend = Trixi._PREFERENCE_THREADING
        Trixi.set_threading_backend!(:kernelabstractions)
        # relaunching julia
        try
            run(`$(Base.julia_cmd()) --threads=$TRIXI_NTHREADS --check-bounds=yes $(abspath("test_kernelabstractions.jl"))`)
        finally
            Trixi.set_threading_backend!(Symbol(previous_backend))
        end
    end
end
