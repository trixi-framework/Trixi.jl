using TestItemRunner

# We run subsets of the test suite in parallel CI jobs by setting the `TRIXI_TEST`
# environment variable. Its value is matched against the `tags` attached to each
# `@testitem`. By default (`TRIXI_TEST == "all"`) we run everything.
const TRIXI_TEST = get(ENV, "TRIXI_TEST", "all")

# TODO (TestItems.jl migration): The suites that need a specially-launched Julia
# process - `mpi` (via `mpiexec -n N`), `threaded` (via `--threads=N`), and
# `kernelabstractions` (relaunch with a different threading backend preference) -
# will be dispatched here: when `TRIXI_TEST` selects one of them, `run()` the
# appropriate launcher, which itself calls a tag-filtered `@run_package_tests`.
# Until those files are migrated, the in-process run below covers the migrated
# `@testitem`s only.

@run_package_tests filter=ti->(TRIXI_TEST == "all" || Symbol(TRIXI_TEST) in ti.tags)

# Common setup shared by all `@testitem`s. Listing `setup=[Setup]` on a test item
# makes the `@test_trixi_include` helper macro (and the packages `test_trixi.jl`
# pulls in) available inside it. `using Trixi`/`using Test` are already provided
# automatically by the default imports of every `@testitem`/`@testsnippet`.
@testsnippet Setup begin
    include("test_trixi.jl")
end
