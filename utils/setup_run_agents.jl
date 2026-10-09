#!/usr/bin/env julia

# Create or update the local Julia projects used by AI coding agents (see `AGENTS.md`):
# - `run_agents/` develops the local version of Trixi.jl and contains all test
#   dependencies (copied from `test/Project.toml`, including compat bounds and
#   preferences).
# - `run_agents/formatter/` contains JuliaFormatter.jl pinned to the version used in CI,
#   since its dependencies are incompatible with those of Trixi.jl.
# This script is idempotent: it only re-creates a project if it is outdated. Run it as
#   julia utils/setup_run_agents.jl
# from any directory; all paths are relative to the root of the repository.

using Pkg, TOML

const REPO_DIR = dirname(@__DIR__)
const RUN_AGENTS_DIR = joinpath(REPO_DIR, "run_agents")
const FORMATTER_DIR = joinpath(RUN_AGENTS_DIR, "formatter")
const FORMATTER_VERSION = "1.0.60"

parse_toml(file) = isfile(file) ? TOML.parsefile(file) : Dict{String, Any}()

function setup_run_agents()
    mkpath(RUN_AGENTS_DIR)
    test_project = TOML.parsefile(joinpath(REPO_DIR, "test", "Project.toml"))
    project_file = joinpath(RUN_AGENTS_DIR, "Project.toml")
    project = parse_toml(project_file)
    manifest = parse_toml(joinpath(RUN_AGENTS_DIR, "Manifest.toml"))

    # Reuse the environment only if it still contains all test dependencies with the
    # same compat bounds etc. and develops this checkout of Trixi.jl
    deps = get(project, "deps", Dict{String, Any}())
    trixi_entries = get(get(manifest, "deps", Dict{String, Any}()), "Trixi", [])
    test_deps_present = issubset(keys(test_project["deps"]), keys(deps))
    sections_match = all(get(project, section, nothing) ==
                         get(test_project, section, nothing)
                         for section in ("compat", "extras", "preferences", "targets"))
    local_trixi = haskey(deps, "Trixi") &&
                  any(haskey(entry, "path") &&
                      samefile(joinpath(RUN_AGENTS_DIR, entry["path"]), REPO_DIR)
                      for entry in trixi_entries)
    up_to_date = test_deps_present && sections_match && local_trixi

    Pkg.activate(RUN_AGENTS_DIR)
    if !up_to_date
        cp(joinpath(REPO_DIR, "test", "Project.toml"), project_file; force = true)
        # A relative path is stored relative to `run_agents/` in the manifest
        Pkg.develop(path = relpath(REPO_DIR))
    end
    Pkg.instantiate()
end

function setup_formatter()
    manifest = parse_toml(joinpath(FORMATTER_DIR, "Manifest.toml"))
    entries = get(get(manifest, "deps", Dict{String, Any}()), "JuliaFormatter", [])
    pinned = any(get(entry, "version", nothing) == FORMATTER_VERSION &&
                 get(entry, "pinned", false) for entry in entries)

    pinned || rm(FORMATTER_DIR; force = true, recursive = true)
    Pkg.activate(FORMATTER_DIR)
    if !pinned
        Pkg.add(name = "JuliaFormatter", version = FORMATTER_VERSION)
        Pkg.pin("JuliaFormatter")
    end
    Pkg.instantiate()
end

setup_run_agents()
setup_formatter()
