using Pkg, Libdl
# Configure the test setup based on environment variables set in CI.
# First, we get the settings and remove all local preference configurations
# that may still exist.
const JULIA_HDF5_PATH = get(ENV, "JULIA_HDF5_PATH", "")
rm(joinpath(dirname(@__DIR__), "LocalPreferences.toml"); force=true)

# Next, we configure MPI.jl appropriately.
import MPI
MPI.MPIPreferences.use_system_binary()

# Finally, we configure HDF5.jl as desired.
import UUIDs
MPI.MPIPreferences.Preferences.set_preferences!(
    UUIDs.UUID("f67ccb44-e63f-5c2f-98bd-6dc0ccc4ba2f"), # UUID of HDF5.jl
    "libhdf5" => joinpath(JULIA_HDF5_PATH, "libhdf5." * Libdl.dlext),
    "libhdf5_hl" => joinpath(JULIA_HDF5_PATH, "libhdf5_hl." * Libdl.dlext);
    force=true
)
Pkg.build("HDF5")