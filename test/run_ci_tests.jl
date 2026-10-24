# Prepare the test environment and run the tests in a CI job, e.g., from the root of the
# repository,
#   TRIXI_TEST=CUDA julia --color=yes test/run_ci_tests.jl CUDA
# The arguments are packages listed in the `[extras]` of `test/Project.toml` that are
# required for the tests, e.g., `CUDA`, `AMDGPU`, or `Metal` for the GPU tests. They are
# not regular test dependencies so that they do not restrict the versions of other
# packages in the test environment for all other CI jobs.
#
# On the oldest Julia version supported by Trixi.jl, all packages are downgraded to the
# oldest versions compatible with the compat bounds. On newer Julia versions, the newest
# compatible versions are used. Set the environment variable `TRIXI_TEST_DOWNGRADE` to
# `true` or `false` to override this choice.
#
# This script modifies `test/Project.toml` and creates `Manifest.toml` files in the root
# directory of the repository and in `test/`. Thus, it refuses to run outside of CI
# (where the environment variable `CI` is `true`) to protect local work.

using Pkg: Pkg
using TOML: TOML
using Downloads: Downloads

const REPO_DIR = dirname(@__DIR__)
const TEST_DIR = joinpath(REPO_DIR, "test")

if get(ENV, "CI", "false") != "true"
    error("This script is intended to be run in CI since it modifies `test/Project.toml` ",
          "and creates `Manifest.toml` files. Set the environment variable `CI=true` ",
          "to run it anyway, e.g., in a separate clone of the repository.")
end

# Move the requested packages from the `[extras]` to the `[deps]` of `test/Project.toml`.
# Their compat bounds are already listed in `test/Project.toml`.
test_project_file = joinpath(TEST_DIR, "Project.toml")
test_project = TOML.parsefile(test_project_file)
for name in ARGS
    if !haskey(test_project["extras"], name)
        error("$name is not listed in the `[extras]` of `test/Project.toml`")
    end
    test_project["deps"][name] = pop!(test_project["extras"], name)
end
# julia-actions/julia-downgrade-compat merges the main and the test project without the
# `[extras]`, so the compat entries of the packages remaining in the `[extras]` would be
# invalid there. Since these packages are not used in this job, we remove their compat
# entries.
for name in keys(test_project["extras"])
    delete!(test_project["compat"], name)
end
open(io -> TOML.print(io, test_project; sorted = true), test_project_file, "w")

# Downgrade the packages on the oldest Julia version supported by Trixi.jl, i.e., the
# lower bound of the Julia compat entry
julia_compat = TOML.parsefile(joinpath(REPO_DIR, "Project.toml"))["compat"]["julia"]
oldest_julia = VersionNumber(strip(first(split(julia_compat, ","))))
on_oldest_julia = (VERSION.major, VERSION.minor) == (oldest_julia.major, oldest_julia.minor)
downgrade = parse(Bool, get(ENV, "TRIXI_TEST_DOWNGRADE", string(on_oldest_julia)))

if downgrade
    @info "Downgrading packages to the oldest compatible versions" VERSION
    # We use the script of the GitHub action julia-actions/julia-downgrade-compat, which
    # runs Resolver.jl to find the oldest compatible versions, so that we can use the same
    # approach on GitHub Actions and Buildkite.
    downgrade_script = Downloads.download("https://raw.githubusercontent.com/julia-actions/julia-downgrade-compat/v2/downgrade.jl")
    # Packages that are not downgraded (e.g., standard libraries)
    skip = "LinearAlgebra,Printf,SparseArrays,UUIDs,DelimitedFiles,Test,Downloads,Random"
    # Use the main project and the test project, verify that all direct dependencies are
    # downgraded to the lower bounds of their compat entries, and resolve for the current
    # Julia version
    cmd = `$(Base.julia_cmd()) --color=yes $downgrade_script $skip ".,test" forcedeps 1`
    run(Cmd(cmd; dir = REPO_DIR))
else
    @info "Using the newest compatible versions of the packages" VERSION
end

# We run the tests manually instead of using `Pkg.test` since `Pkg.test` does not use the
# downgraded `Manifest.toml` under Julia < v1.12, see also
# https://github.com/julia-actions/julia-downgrade-compat/blob/00f940b7be2b50389571ed016d603be561649103/README.md#L89
Pkg.activate(TEST_DIR)
Pkg.develop(Pkg.PackageSpec(path = REPO_DIR))
Pkg.instantiate()
Pkg.status(; mode = Pkg.PKGMODE_MANIFEST)

# Use the same Julia options as `Pkg.test`
cmd = ```
      $(Base.julia_cmd()) --project=$(TEST_DIR) --color=yes
      --check-bounds=yes --warn-overwrite=yes --depwarn=yes
      $(joinpath(TEST_DIR, "runtests.jl"))
      ```
process = run(ignorestatus(cmd))
exit(process.exitcode)
