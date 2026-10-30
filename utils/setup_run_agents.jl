#!/usr/bin/env julia

# Create or update the local Julia projects used by AI coding agents (see `AGENTS.md`):
# - `run_agents/` develops the local version of Trixi.jl and contains all test
#   dependencies (copied from `test/Project.toml`, including compat bounds and
#   preferences).
# - `run_agents/kernelabstractions/` is a copy of `run_agents/` with the threading backend
#   of Trixi.jl set to KernelAbstractions.jl, as in the `kernelabstractions` CI jobs.
# - `run_agents/formatter/` contains JuliaFormatter.jl pinned to the version used in CI,
#   since its dependencies are incompatible with those of Trixi.jl.
# This script is idempotent: it only re-creates a project if it is outdated. Run it as
#   julia utils/setup_run_agents.jl
# from any directory; all paths are relative to the root of the repository.
#
# Optionally, pass another directory for the project with the test dependencies, e.g.,
#   julia +release utils/setup_run_agents.jl run_kaimon
# This is useful for a Julia session with a different Julia version than `run_agents/`,
# since a manifest should only be used with the Julia version it was resolved for. In
# this case, the KernelAbstractions.jl and formatter projects are not set up, and an
# existing project in this directory is never re-created or resolved with another Julia
# version, since it may contain additional packages added by the developer. Instead, the
# script throws an error explaining what is outdated.

using Pkg, TOML

# Precompile packages lazily when they are loaded in tests instead of precompiling the
# complete manifest, which takes long (e.g., for GPU packages) and fails completely if
# the precompilation of a single package (not even loaded in the tests) fails
ENV["JULIA_PKG_PRECOMPILE_AUTO"] = "0"

const REPO_DIR = dirname(@__DIR__)
const CUSTOM_DIR = !isempty(ARGS)
const RUN_AGENTS_DIR = joinpath(REPO_DIR, CUSTOM_DIR ? only(ARGS) : "run_agents")
const KERNELABSTRACTIONS_DIR = joinpath(REPO_DIR, "run_agents", "kernelabstractions")
const FORMATTER_DIR = joinpath(REPO_DIR, "run_agents", "formatter")
const FORMATTER_VERSION = "1.0.60"

parse_toml(file) = isfile(file) ? TOML.parsefile(file) : Dict{String, Any}()

# Check whether all entries of `required` are contained in `actual`, recursively for
# nested tables such as `[preferences.HostCPUFeatures]`. Additional entries are allowed,
# e.g., compat bounds of packages or preferences added by developers.
contains_entries(actual, required) = actual == required
function contains_entries(actual::AbstractDict, required::AbstractDict)
    return all(haskey(actual, key) && contains_entries(actual[key], value)
               for (key, value) in required)
end

function contains_section(project, required, section)
    return contains_entries(get(project, section, Dict{String, Any}()),
                            get(required, section, Dict{String, Any}()))
end

function setup_run_agents()
    mkpath(RUN_AGENTS_DIR)
    test_project = TOML.parsefile(joinpath(REPO_DIR, "test", "Project.toml"))
    project_file = joinpath(RUN_AGENTS_DIR, "Project.toml")
    project = parse_toml(project_file)
    manifest = parse_toml(joinpath(RUN_AGENTS_DIR, "Manifest.toml"))

    # Reuse the environment only if it still contains all test dependencies with the
    # same compat bounds etc. and develops this checkout of Trixi.jl. Additional
    # dependencies and corresponding entries are allowed.
    deps = get(project, "deps", Dict{String, Any}())
    trixi_entries = get(get(manifest, "deps", Dict{String, Any}()), "Trixi", [])
    test_deps_present = issubset(keys(test_project["deps"]), keys(deps))
    sections_match = all(contains_section(project, test_project, section)
                         for section in ("compat", "extras", "preferences", "targets"))
    local_trixi = haskey(deps, "Trixi") &&
                  any(haskey(entry, "path") &&
                      samefile(joinpath(RUN_AGENTS_DIR, entry["path"]), REPO_DIR)
                      for entry in trixi_entries)
    up_to_date = test_deps_present && sections_match && local_trixi

    if CUSTOM_DIR && isfile(project_file)
        manifest_version = get(manifest, "julia_version", nothing)
        if manifest_version !== nothing &&
           Base.thisminor(VersionNumber(manifest_version)) != Base.thisminor(VERSION)
            error("The manifest in $RUN_AGENTS_DIR was resolved with Julia ",
                  "$manifest_version, but this is Julia $VERSION. ",
                  "Run this script with the same Julia version.")
        end
        if !up_to_date
            error("The project in $RUN_AGENTS_DIR is outdated (test dependencies ",
                  "present: $test_deps_present, compat etc. match test/Project.toml: ",
                  "$sections_match, develops this checkout of Trixi.jl: $local_trixi). ",
                  "Update it manually or remove the directory and run this script again.")
        end
    end

    Pkg.activate(RUN_AGENTS_DIR)
    if !up_to_date
        cp(joinpath(REPO_DIR, "test", "Project.toml"), project_file; force = true)
        # A relative path is stored relative to the project in the manifest
        Pkg.develop(path = relpath(REPO_DIR))
    end
    Pkg.instantiate()
end

# The project and manifest are copied from `run_agents/` so that both environments use
# the same package versions and share most precompiled packages. Changes made directly
# in `run_agents/kernelabstractions/` are overwritten whenever `run_agents/` differs.
function setup_kernelabstractions()
    mkpath(KERNELABSTRACTIONS_DIR)
    project = parse_toml(joinpath(RUN_AGENTS_DIR, "Project.toml"))
    manifest = parse_toml(joinpath(RUN_AGENTS_DIR, "Manifest.toml"))
    # Relative paths of developed packages (such as Trixi.jl) are stored relative to the
    # manifest
    for entries in values(manifest["deps"]), entry in entries
        path = get(entry, "path", nothing)
        if path !== nothing && !isabspath(path)
            entry["path"] = relpath(joinpath(RUN_AGENTS_DIR, path), KERNELABSTRACTIONS_DIR)
        end
    end

    preferences_file = joinpath(KERNELABSTRACTIONS_DIR, "LocalPreferences.toml")
    preferences = parse_toml(preferences_file)
    get!(preferences, "Trixi", Dict{String, Any}())["backend"] = "kernelabstractions"

    for (file, content) in (joinpath(KERNELABSTRACTIONS_DIR, "Project.toml") => project,
                            joinpath(KERNELABSTRACTIONS_DIR, "Manifest.toml") => manifest,
                            preferences_file => preferences)
        if parse_toml(file) != content
            open(io -> TOML.print(io, content; sorted = true), file, "w")
        end
    end
    Pkg.activate(KERNELABSTRACTIONS_DIR)
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
if isempty(ARGS)
    setup_kernelabstractions()
    setup_formatter()
end
