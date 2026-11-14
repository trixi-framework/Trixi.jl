# Creates the test environment of this checkout under `run_hpc/<version>/test`.
# It holds the dependencies of `test/Project.toml` and `hpc/Project.toml` and develops Trixi.

using Pkg, TOML

const root = dirname(dirname(@__DIR__))
const project = joinpath(root, "run_hpc", "$(VERSION.major).$(VERSION.minor)", "test")

function write_toml(path, data)
    open(path, "w") do io
        TOML.print(io, data; sorted = true)
    end
end

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
