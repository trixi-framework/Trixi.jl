using TestItemRunner

# We run subsets of the test suite in parallel CI jobs by setting the `TRIXI_TEST`
# environment variable. Its value is matched against the `tags` attached to each
# `@testitem`. By default (`TRIXI_TEST == "threaded"`) we run a
# selection of meaningful tests that cover a broad part of the code.
const TRIXI_TEST = get(ENV, "TRIXI_TEST", "threaded")

# Some GitHub CI runners may not have much RAM and just 3 virtual CPU cores.
# In this case, we do not want to use all of the cores to speed-up CI.
const TRIXI_MPI_NPROCS = clamp(Sys.CPU_THREADS - 1, 2, 3)
const TRIXI_NTHREADS = clamp(Sys.CPU_THREADS, 2, 3)

# A few suites cannot run in the ordinary in-process `@run_package_tests` model:
# they need a specially-launched Julia process - `mpi` (multiple ranks via
# `mpiexec`), `threaded` (multiple threads via `--threads`), and
# `kernelabstractions` (a different threading backend, selected via a preference
# that only takes effect on a fresh Julia start). We handle them by *relaunching*
# Julia/`mpiexec` on this very file: the launched worker re-enters `runtests.jl`
# with `TRIXI_TEST_RUN_ITEMS` set and then runs the tag-filtered test items
# in-process (`TestItemRunner` evaluates items in the current process, so this
# also works for every MPI rank). The `TRIXI_TEST` value of such a suite selects
# the items via the equally-named tag; `threaded_legacy` reuses the `threaded`
# items but is launched on a different Julia version by CI.
const SPECIAL_PROCESS_SUITES = ("mpi", "threaded", "kernelabstractions")
const IN_WORKER = haskey(ENV, "TRIXI_TEST_RUN_ITEMS")

# Remove Trixi's output directory `out`, where examples write solution/restart/mesh
# files. We do this once at the start of a run *and* register it to run again when
# the process exits (`atexit`, so it also fires when a test set fails and throws),
# leaving a clean working tree behind. Only the parent process handles this: workers
# share the working directory and finish before the parent exits, so a single
# cleanup here avoids races on `out`.
function clean_outdir()
    isdir("out") && rm("out", recursive = true, force = true)
end
if !IN_WORKER
    clean_outdir()
    atexit(clean_outdir)
end

target_tag(suite) = suite == "threaded_legacy" ? :threaded : Symbol(suite)

# Special suites this (parent) invocation will dispatch into their own processes.
const SUITES_TO_DISPATCH = if IN_WORKER
    String[]
elseif TRIXI_TEST == "all"
    collect(SPECIAL_PROCESS_SUITES)
elseif TRIXI_TEST in SPECIAL_PROCESS_SUITES || TRIXI_TEST == "threaded_legacy"
    [TRIXI_TEST]
else
    String[]
end

# `import` is only allowed at top level, so load here what `dispatch_special_suite`
# needs: `MPI` for `mpiexec`, and `Trixi` to toggle the threading backend preference.
if "mpi" in SUITES_TO_DISPATCH
    import MPI
end
if "kernelabstractions" in SUITES_TO_DISPATCH
    import Trixi
end

# The GPU suites run their items only when the respective backend is `functional()`,
# so requesting `CUDA`/`AMDGPU` on a machine without that GPU warns and runs nothing
# instead of erroring.
RUN_CUDA = false
if !IN_WORKER && TRIXI_TEST in ("all", "CUDA")
    import CUDA
    RUN_CUDA = CUDA.functional()
    RUN_CUDA || @warn "Unable to run CUDA tests on this machine"
end
RUN_AMDGPU = false
if !IN_WORKER && TRIXI_TEST in ("all", "AMDGPU")
    import AMDGPU
    RUN_AMDGPU = AMDGPU.functional()
    RUN_AMDGPU || @warn "Unable to run AMDGPU tests on this machine"
end

# Relaunch Julia/`mpiexec` for a suite that needs a special process. The worker
# re-enters this file with `TRIXI_TEST=<suite>` and `TRIXI_TEST_RUN_ITEMS=true`.
function run_worker(cmd, suite)
    run(addenv(cmd, "TRIXI_TEST_RUN_ITEMS" => "true", "TRIXI_TEST" => suite))
end

function dispatch_special_suite(suite)
    project = dirname(Base.active_project())
    julia = Base.julia_cmd()

    if suite == "mpi"
        cmd = `$(MPI.mpiexec()) -n $TRIXI_MPI_NPROCS $julia --threads=1 --check-bounds=yes --heap-size-hint=0.5G --project=$project $(@__FILE__)`
        run_worker(cmd, suite)
    elseif suite == "threaded" || suite == "threaded_legacy"
        cmd = `$julia --threads=$TRIXI_NTHREADS --check-bounds=yes --code-coverage=none --project=$project $(@__FILE__)`
        run_worker(cmd, suite)
    elseif suite == "kernelabstractions"
        # The threading backend is selected via a preference that is read on Julia
        # startup, so we set it here (in the parent) and restore it afterwards;
        # the relaunched worker picks it up.
        previous_backend = Trixi._PREFERENCE_THREADING
        Trixi.set_threading_backend!(:kernelabstractions)
        try
            cmd = `$julia --threads=$TRIXI_NTHREADS --check-bounds=yes --project=$project $(@__FILE__)`
            run_worker(cmd, suite)
        finally
            Trixi.set_threading_backend!(Symbol(previous_backend))
        end
    end
end

if !isempty(SUITES_TO_DISPATCH)
    # Dispatch each requested special suite into its own process; the worker(s)
    # run the tagged items. For a single special suite that is all we do; for
    # `all` we additionally run the remaining (in-process) items below.
    foreach(dispatch_special_suite, SUITES_TO_DISPATCH)
end

if !IN_WORKER && (TRIXI_TEST in SPECIAL_PROCESS_SUITES || TRIXI_TEST == "threaded_legacy")
    # A single special suite was requested and dispatched above; nothing to run
    # in this process.
else
    # In-process run. Either we are inside a launched worker (run just that suite's
    # tagged items), or this is an ordinary partition / `all` (in which case we
    # exclude the special suites, which must run in their own processes).
    CI_ON_WINDOWS = (get(ENV, "GITHUB_ACTIONS", "false") == "true") && Sys.iswindows()
    special_tags = (:mpi, :threaded, :kernelabstractions)
    tag = target_tag(TRIXI_TEST)

    @run_package_tests filter = ti -> begin
        if TRIXI_TEST == "all"
            # The special suites run in dedicated processes (see above).
            any(t -> t in ti.tags, special_tags) && return false
        else
            tag in ti.tags || return false
        end
        # CI with MPI fails often on Windows for some examples; skip those there.
        if CI_ON_WINDOWS && :mpi_skip_windows in ti.tags
            return false
        end
        # GPU items run only when the respective backend is functional.
        :CUDA in ti.tags && !RUN_CUDA && return false
        :AMDGPU in ti.tags && !RUN_AMDGPU && return false
        return true
    end
end

# Common setup shared by all `@testitem`s. Listing `setup=[Setup]` on a test item
# makes the `@test_trixi_include` helper macro (and the packages `test_trixi.jl`
# pulls in) available inside it. `using Trixi`/`using Test` are already provided
# automatically by the default imports of every `@testitem`/`@testsnippet`.
@testsnippet Setup begin
    include("test_trixi.jl")
end
