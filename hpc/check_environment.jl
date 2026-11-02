using TOML, UUIDs

"$(VERSION.major).$(VERSION.minor)" == only(ARGS) ||
    error("The Julia executable does not match the requested version.")
project = TOML.parsefile(joinpath(@__DIR__, "Project.toml"))
preferences = project["preferences"]
for (name, uuid) in project["extras"]
    effective = Base.get_preferences(UUID(uuid))
    for (key, value) in preferences[name]
        @assert get(effective, key, nothing)==value "Unexpected preference for $name.$key."
    end
end

using Trixi, CUDA, MPI, Libdl, TestItemRunner

MPI.versioninfo()
@assert MPI.MPIPreferences.binary == "system"
@assert MPI.MPIPreferences.abi == "OpenMPI"
@assert realpath(Libdl.dlpath(MPI.libmpi)) ==
        realpath(preferences["MPIPreferences"]["libmpi"])
@assert occursin("Open MPI", MPI.Get_library_version())
@assert Trixi.HDF5.has_parallel()
@assert Trixi.P4est.uses_mpi()
for (package, library, actual) in (("P4est", "libp4est", Trixi.P4est.LibP4est.libp4est),
                                   ("P4est", "libsc", Trixi.P4est.LibP4est.libsc),
                                   ("T8code", "libt8", Trixi.T8code.Libt8.libt8),
                                   ("T8code", "libp4est", Trixi.T8code.Libt8.libp4est),
                                   ("T8code", "libsc", Trixi.T8code.Libt8.libsc),
                                   ("HDF5", "libhdf5", Trixi.HDF5.API.libhdf5),
                                   ("HDF5", "libhdf5_hl", Trixi.HDF5.API.libhdf5_hl))
    @assert realpath(Libdl.dlpath(Libdl.dlopen(actual))) ==
            realpath(preferences[package][library])
end
mpi_libraries = filter(path -> startswith(basename(path), "libmpi.so"), Libdl.dllist())
@assert length(unique(realpath.(mpi_libraries))) == 1

mktempdir(ENV["TMPDIR"]) do directory
    filename = joinpath(directory, "parallel.h5")
    Trixi.HDF5.h5open(filename, "w", MPI.COMM_SELF) do file
        file["values"] = [1, 2, 3]
    end
    Trixi.HDF5.h5open(filename, "r", MPI.COMM_SELF) do file
        @assert read(file["values"]) == [1, 2, 3]
    end
end

@assert Base.get_extension(Trixi, :TrixiCUDACoreExt) !== nothing
toolkit = VersionNumber(preferences["CUDA_Runtime_jll"]["version"])
@assert VersionNumber(preferences["CUDA_Compiler_jll"]["version"]) == toolkit
@assert Base.thisminor(CUDA.runtime_version()) == toolkit
@assert Base.thisminor(VersionNumber(ENV["EBVERSIONCUDA"])) == toolkit
@info "Loaded mesh libraries" p4est=pkgversion(Trixi.P4est) t8code=pkgversion(Trixi.T8code)
@info "Trixi threading backend" backend=Trixi._PREFERENCE_THREADING
CUDA.precompile_runtime()

names = Set(("P4estMesh2D: elixir_advection_basic.jl",
             "T8codeMesh2D: elixir_advection_basic.jl"))
selected = Set{String}()
cd(joinpath(dirname(@__DIR__), "test")) do
    TestItemRunner.run_tests(pwd();
                             filter = ti -> begin
                                 ti.name in names || return false
                                 push!(selected, ti.name)
                                 return true
                             end,
                             verbose = true)
end
@assert selected == names
