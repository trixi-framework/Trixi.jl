# HPC setup

The Otus configuration supports Julia 1.10 and 1.13 with system OpenMPI, parallel HDF5 and local CUDA 12.8.
The environment script and [preferences project](Project.toml) define one fixed toolchain and must be updated together.
The [system MPI documentation](../docs/src/parallelization.md) explains the library requirements.

Set `CI_DEPOT_ROOT` to an absolute shared depot directory.
Load the environment and install the pinned native libraries from Bash.

```bash
export CI_DEPOT_ROOT=/scratch/hpc-prf-cobentx/depot
source hpc/env.sh 1.13 &&
    bash hpc/install_libraries.sh
```

[env.sh](env.sh) loads the Otus modules and sets explicit Julia depot and load paths that exclude the user depot.
Cluster preferences must be on the load path before dependency resolution or package loading.
Pass `-C "$JULIA_CPU_TARGET"` to Julia so runtime compilation matches package compilation.
CPU targets are fixed for each Julia version.
`CMAKE_BUILD_PARALLEL_LEVEL`, `JULIA_NUM_PRECOMPILE_TASKS`, `JULIA_NUM_THREADS` and `OPENBLAS_NUM_THREADS` control concurrency.

[install_libraries.sh](install_libraries.sh) builds the pinned [CMake recipe](libraries/CMakeLists.txt) under a shared lock and publishes a completion marker after checking the installed libraries.
The separate p4est 2.8 build preserves library versions distinct from t8code's dependencies.
CMake is invoked directly because loading its module would override Julia's libcurl.
Both Julia versions share `/scratch/hpc-prf-cobentx/depot/native/otus-foss-2025a-v2`.
An incompatible installation is rejected, native updates require a new versioned path in the installer and preferences.
The installer computes its fingerprint from the native recipe, OpenMPI, zlib and CMake paths.
Use `bash hpc/install_libraries.sh --print-fingerprint` to record it without modifying the installation.

[check_environment.jl](check_environment.jl) verifies effective preferences, loaded native libraries, parallel HDF5 and driverless CUDA precompilation.
It runs only the existing basic p4est and t8code advection items, which generate their own meshes.
Use an existing test environment that develops this checkout and contains the dependencies from `test/Project.toml`.
For an environment at `run_hpc/1.13/test`, run the following command from the repository root.

```bash
export CI_DEPOT_ROOT=/scratch/hpc-prf-cobentx/depot
source hpc/env.sh 1.13 &&
    julia --startup-file=no -C "$JULIA_CPU_TARGET" --check-bounds=yes \
        --threads=1 --project=run_hpc/1.13/test hpc/check_environment.jl 1.13
```

Repeat validation with `1.10` and its matching environment.
These focused checks run on the login node, complete suites and GPU execution require CI or allocations submitted by the user.
