# Testing

During the development of Trixi.jl, we rely on
[continuous testing](https://en.wikipedia.org/wiki/Continuous_testing) to ensure
that modifications or new features do not break existing
functionality or add other errors. In the main
[Trixi.jl](https://github.com/trixi-framework/Trixi.jl) repository (and the
repositories for the visualization tool
[Trixi2Vtk](https://github.com/trixi-framework/Trixi.jl)), this is facilitated by
[GitHub Actions](https://docs.github.com/en/free-pro-team@latest/actions),
which allows to run tests automatically upon certain events. When, how, and what
is tested by GitHub Actions is controlled by the workflow file
[`.github/workflows/ci.yml`](https://github.com/trixi-framework/Trixi.jl/blob/main/.github/workflows/ci.yml).
In Trixi.jl and its related repositories, tests are triggered by
* each `git push` to `main` and
* each `git push` to any pull request.
Besides checking functionality, we also analyse the [Test coverage](@ref) to
ensure that we do not miss important parts during testing.

!!! note "Test and coverage requirements"
    Before merging a pull request (PR) to `main`, we require that
    * the code passes all functional tests
    * code coverage does not decrease.


## Testing setup
The entry point for all testing is the file
[`test/runtests.jl`](https://github.com/trixi-framework/Trixi.jl/blob/main/test/runtests.jl),
which is run by the automated tests and which can be triggered manually by
executing
```julia
julia> using Pkg; Pkg.test("Trixi")
```
in the REPL.

Trixi.jl's tests are organized with
[TestItems.jl](https://github.com/julia-vscode/TestItems.jl) and executed with
[TestItemRunner.jl](https://github.com/julia-vscode/TestItemRunner.jl). Every individual
test is a self-contained `@testitem` block, and these are collected in the various files
named `test_xxx.jl` in the `test` directory (e.g., all 2D tests on the `P4estMesh` live
in
[`test/test_p4est_2d.jl`](https://github.com/trixi-framework/Trixi.jl/blob/main/test/test_p4est_2d.jl)).
Because every `@testitem` is independent and discoverable on its own, you can run a
single test or a small subset without running the whole suite — either directly from the
[Testing UI of the Julia VS Code extension](https://www.julia-vscode.org/docs/stable/userguide/testitems/)
or from the REPL as shown below.

To run a subset from the REPL, first activate the test environment (which merges the
package with its test-only dependencies), for example using
[TestEnv.jl](https://github.com/JuliaTesting/TestEnv.jl), and then call
`@run_package_tests` with a `filter` on the test item's `name`, `tags`, or `filename`:
```julia
julia> using TestEnv; TestEnv.activate()

julia> using TestItemRunner

julia> cd("test")

julia> @run_package_tests filter = ti -> occursin("TreeMesh2D Advection: elixir_advection_basic.jl", ti.name)

julia> # Run every test tagged `:tree_part1`
       @run_package_tests filter = ti -> :tree_part1 in ti.tags
```

For the automated tests with GitHub Actions, we run multiple jobs in parallel to reduce
the waiting time until all tests are finished. Each job runs a subset of the tests,
selected via the `TRIXI_TEST` environment variable whose value is matched against the
`tags` attached to each `@testitem`. You can reproduce a specific job locally by setting
`TRIXI_TEST` accordingly, e.g., from the shell

```bash
TRIXI_TEST=tree_part1 julia --project=. -e 'using Pkg; Pkg.test("Trixi")'
```

You can see the different components that are run as jobs by looking at the `TRIXI_TEST`
values in
[`test/runtests.jl`](https://github.com/trixi-framework/Trixi.jl/blob/main/test/runtests.jl)
and
[`.github/workflows/ci.yml`](https://github.com/trixi-framework/Trixi.jl/blob/main/.github/workflows/ci.yml).


### GPU tests
The tests for the GPU backends cannot run on GitHub Actions since they require actual
hardware. They are therefore executed on [Buildkite](https://buildkite.com) on dedicated
machines with NVIDIA (`TRIXI_TEST=CUDA`) and AMD (`TRIXI_TEST=AMDGPU`) GPUs, configured in
[`.buildkite/pipeline.yml`](https://github.com/trixi-framework/Trixi.jl/blob/main/.buildkite/pipeline.yml).

Since most pull requests do not touch any GPU-related code and the GPU machines are a
scarce resource, these tests do **not** run automatically. Instead, they are only run
* on demand, by writing a comment containing
  ```
  /run_gpu_tests
  ```
  on the pull request. This runs both the CUDA and the AMDGPU tests. The comment must be
  written by an owner, member, or collaborator of the repository,
* automatically for every push to `main`, i.e., after a pull request has been merged, and
* when a build is started manually from the Buildkite web interface.

The comment is picked up by
[`.github/workflows/TriggerGPUTests.yml`](https://github.com/trixi-framework/Trixi.jl/blob/main/.github/workflows/TriggerGPUTests.yml),
which asks Buildkite to build the *current* head commit of the pull request. Hence, you
need to comment again after pushing further changes. If you modify GPU code, please
request a GPU run before merging.


## Adding new tests
We use [TestItems.jl](https://github.com/julia-vscode/TestItems.jl) on top of Julia's
built-in [unit testing capabilities](https://docs.julialang.org/en/v1/stdlib/Test/):
each test is a `@testitem` block whose body uses the usual `@test` assertions. In
general, newly added code must be covered by at least one test, and all new elixirs added
to the `examples/` directory must be used at least once during testing. New tests should
be added as a `@testitem` to the corresponding `test/test_xxx.jl` file, e.g., a test
involving the 3D linear advection equation on the `TreeMesh` would go into
[`test/test_tree_3d_advection.jl`](https://github.com/trixi-framework/Trixi.jl/blob/main/test/test_tree_3d_advection.jl).
Each `@testitem` lists the setup snippets it needs via `setup = [Setup, ...]` and the CI
job(s) it belongs to via `tags = [...]` (see [Testing setup](@ref) for how `tags`
correspond to the `TRIXI_TEST` jobs). Please study one of the existing tests and stay
consistent to the current style when creating new tests.

Since we want to test as much as possible, we have a lot of tests and
frequently create new ones. Naturally, this increases the time to wait for all
tests to pass with each novel feature added to Trixi.jl. Therefore, new tests should be as
short as reasonably possible, i.e., without being too insensitive to pick up
changes or errors in the code.

When you add new tests, please check whether all CI jobs still take approximately
the same time. If the job where you added new tests takes much longer than
everything else, please consider moving some tests from one job to another
(or report this incident and ask the main developers for help).

!!! note "Test duration"
    As a general rule, tests should last **no more than 10 seconds** when run
    with a single thread and after compilation (i.e., excluding the first run).


## Test coverage
In addition to ensuring that the code produces the expected results, the
automated tests also record the
[code coverage](https://en.wikipedia.org/wiki/Code_coverage). The resulting
coverage reports, i.e., which lines of code were executed by at least one test
and are thus considered "covered" by testing, are automatically uploaded to
[Coveralls](https://coveralls.io) for easy analysis. Typically, you see a number
of Coveralls results at the bottom of each pull request: One for each parallel
job (see [Testing setup](@ref)), which can usually be ignored since they only
cover parts of the code by definition, and a cumulative coverage result named
`coverage/coveralls`. The "Details" link takes you to a detailed report on
which lines of code are covered by tests, which ones are missed, and especially
which *new* lines the pull requests adds to Trixi.jl's code base that are not yet
covered by testing.
!!! note "Coverage requirements"
    In general, we require pull requests to *not decrease* the overall
    test coverage percentage in `main`, with a **hard lower bound of 97%**.
