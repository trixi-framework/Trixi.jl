using Pkg, Libdl
Pkg.activate(dirname(@__DIR__))
Pkg.instantiate()
# Configure the test setup based on environment variables set in CI.
# First, we get the settings and remove all local preference configurations
# that may still exist.
run(`sudo apt-get update`)
run(`sudo apt-get install mpich libhdf5-mpich-dev`)
run(`echo "JULIA_HDF5_PATH=/usr/lib/x86_64-linux-gnu/hdf5/mpich/" \>\> $GITHUB_ENV`)
const JULIA_HDF5_PATH = get(ENV, "JULIA_HDF5_PATH", "")
rm(joinpath(dirname(@__DIR__), "LocalPreferences.toml"); force=true)

# Next, we configure MPI.jl appropriately.
using MPIPreferences
MPIPreferences.use_system_binary()

# Finally, we configure HDF5.jl as desired.
import UUIDs
MPIPreferences.Preferences.set_preferences!(
    UUIDs.UUID("f67ccb44-e63f-5c2f-98bd-6dc0ccc4ba2f"), # UUID of HDF5.jl
    "libhdf5" => joinpath(JULIA_HDF5_PATH, "libhdf5." * Libdl.dlext),
    "libhdf5_hl" => joinpath(JULIA_HDF5_PATH, "libhdf5_hl." * Libdl.dlext);
    force=true
)
Pkg.add("HDF5")
Pkg.resolve()