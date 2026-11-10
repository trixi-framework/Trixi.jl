using SHA, TOML

const root = ENV["CI_PROJECT_DIR"]
const prepared = joinpath(root, "run_hpc", ENV["JULIA_VERSION"])
const project = joinpath(prepared, "test")
const mode = first(ARGS)
@assert length(ARGS) == (mode == "backend" ? 2 : 1)
@assert "$(VERSION.major).$(VERSION.minor)" == ENV["JULIA_VERSION"]
# Benchmarks run with automatic bounds checking, so precompilation also covers that mode.
@assert Base.JLOptions().check_bounds == 1 ||
        (mode == "precompile" && Base.JLOptions().check_bounds == 0)
@assert unsafe_string(Base.JLOptions().cpu_target) == ENV["JULIA_CPU_TARGET"]
@assert readchomp(`git -C $root rev-parse HEAD`) == ENV["CI_COMMIT_SHA"]

file_hash(path) = bytes2hex(open(sha256, path))
function write_toml(path, data)
    open(path, "w") do io
        TOML.print(io, data; sorted = true)
    end
end
function provenance()
    preferences = TOML.parsefile(joinpath(root, "hpc", "Project.toml"))["preferences"]
    native = dirname(dirname(dirname(preferences["P4est"]["libp4est"])))
    return Dict("pipeline" => ENV["CI_PIPELINE_ID"], "commit" => ENV["CI_COMMIT_SHA"],
                "julia" => string(VERSION), "cpu_target" => ENV["JULIA_CPU_TARGET"],
                "native" => readchomp(joinpath(native, "VERSION")),
                "environment" => file_hash(joinpath(root, "hpc", "env.sh")),
                "preferences" => file_hash(joinpath(root, "hpc", "Project.toml")),
                "files" => Dict(name => file_hash(joinpath(project, name))
                                for name in ("Project.toml", "Manifest.toml",
                                             "LocalPreferences.toml")))
end

if mode == "prepare"
    using Pkg
    mkpath(project)
    data = TOML.parsefile(joinpath(root, "test", "Project.toml"))
    hpc = TOML.parsefile(joinpath(root, "hpc", "Project.toml"))
    for section in ("deps", "compat", "extras")
        merge!(get!(data, section, Dict{String, Any}()), hpc[section])
    end
    write_toml(joinpath(project, "Project.toml"), data)
    rm(joinpath(project, "Manifest.toml"); force = true)
    write_toml(joinpath(project, "LocalPreferences.toml"),
               Dict("Trixi" => Dict("backend" => "polyester")))
    isempty(Pkg.Registry.reachable_registries()) ? Pkg.Registry.add("General") :
    Pkg.Registry.update()
    cd(project) do
        Pkg.activate(".")
        Pkg.develop(path = relpath(root, project))
        Pkg.instantiate()
    end
    exit()
elseif mode == "freeze"
    write_toml(joinpath(prepared, "provenance.toml"), provenance())
    exit()
elseif mode == "verify"
    @assert realpath(dirname(Base.active_project())) == realpath(project)
    @assert TOML.parsefile(joinpath(prepared, "provenance.toml")) == provenance()
    manifest = TOML.parsefile(joinpath(project, "Manifest.toml"))
    @assert manifest["julia_version"] == string(VERSION)
    path = only(manifest["deps"]["Trixi"])["path"]
    @assert !isabspath(path) && realpath(joinpath(project, path)) == realpath(root)
    cp(joinpath(prepared, "provenance.toml"),
       joinpath(ENV["RESULTS"], "provenance.toml"); force = true)
    exit()
elseif mode == "backend"
    using Trixi
    Trixi.set_threading_backend!(Symbol(ARGS[2]))
elseif mode == "precompile"
    using Pkg
    Pkg.precompile()
else
    error("Unknown environment operation: $mode.")
end
