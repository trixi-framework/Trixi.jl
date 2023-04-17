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
in the REPL. Since there already exist many tests, we have split them up into
multiple files in the `test` directory to allow for faster testing of individual
parts of the code.
Thus in addition to performing all tests, you can also just `include` one of the
files named `test_xxx.jl` to run only a specific subset, e.g.,
```julia
julia> # Run all 2D tests on the P4estMesh
       include(joinpath("test", "test_p4est_2d.jl"))

julia> # Run all 1D tests for the Euler equations on the TreeMesh
       include(joinpath("test", "test_tree_1d_euler.jl"))
```
For the automated tests with GitHub Actions, we run multiple jobs in parallel to
reduce the waiting time until all tests are finished. You can see the different
components that are run as jobs by looking at the `TRIXI_TEST` variable in
[`test/runtests.jl`](https://github.com/trixi-framework/Trixi.jl/blob/main/test/runtests.jl).


## Adding new tests
We use Julia's built-in [unit testing capabilities](https://docs.julialang.org/en/v1/stdlib/Test/)
to configure tests. In general, newly added code must be covered by at least one
test, and all new elixirs added to the `examples/` directory must be used at
least once during testing. New tests should be added to the corresponding
`test/test_xxx.jl` file, e.g., a test involving the 3D linear advection equation
on the `TreeMesh` would go into
[`test/test_tree_3d_advection.jl`](https://github.com/trixi-framework/Trixi.jl/blob/main/test/test_tree_3d_advection.jl).
Please study one of the existing tests and stay consistent to the current style
when creating new tests.

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
