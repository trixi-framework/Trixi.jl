# HPC setup

The Otus setup supports Julia 1.10 and 1.13 with system OpenMPI, parallel HDF5 and local CUDA 12.8.
[env.sh](env.sh) and the [preferences project](Project.toml) define one fixed toolchain and must be updated together.
See the [system MPI documentation](../docs/src/parallelization.md) for library requirements.

Set `CI_DEPOT_ROOT` to an absolute shared depot directory.
Load the environment and install the pinned native libraries from Bash.

```bash
export CI_DEPOT_ROOT=/scratch/hpc-prf-cobentx/depot
source hpc/env.sh 1.13 &&
    bash hpc/install_libraries.sh
```

[env.sh](env.sh) loads the modules and sets Julia depot and load paths that exclude the user depot and expose cluster preferences before resolution and package loading.
Pass `-C "$JULIA_CPU_TARGET"` to Julia so runtime compilation matches package compilation.
OpenMPI selects UCX for CPU and GPU communication.
`UCX_MEMTYPE_CACHE=no` disables cached pointer classification and must be set before Julia starts, following the [MPI.jl compatibility guidance](https://juliaparallel.org/MPI.jl/stable/knownissues/#Memory-cache).
Login node temporary files stay in the checkout, compute jobs preserve the allocation's `TMPDIR` or use `/tmp`.
`CMAKE_BUILD_PARALLEL_LEVEL`, `JULIA_NUM_PRECOMPILE_TASKS`, `JULIA_NUM_THREADS` and `OPENBLAS_NUM_THREADS` control concurrency.

[install_libraries.sh](install_libraries.sh) builds the pinned [CMake recipe](libraries/CMakeLists.txt) under a shared lock and publishes its marker after checking the installed libraries.
The separate p4est 2.8 build preserves library versions distinct from t8code's dependencies.
CMake is invoked directly because loading its module would override Julia's libcurl.
Both Julia versions share `/scratch/hpc-prf-cobentx/depot/native/otus-foss-2025a-v2`.
The installation fingerprint covers the native recipe and OpenMPI, zlib and CMake paths, incompatible installations are rejected.
Native updates require a new versioned path in the installer and preferences.
Use `bash hpc/install_libraries.sh --print-fingerprint` to record it without modifying the installation.

[check_environment.jl](check_environment.jl) verifies effective preferences, loaded native libraries, parallel HDF5 and driverless CUDA precompilation.
It runs the existing basic p4est and t8code advection items, which generate their own meshes.
Use a test environment that develops this checkout and contains `test/Project.toml` dependencies, for example `run_hpc/1.13/test`.

```bash
export CI_DEPOT_ROOT=/scratch/hpc-prf-cobentx/depot
source hpc/env.sh 1.13 &&
    julia --startup-file=no -C "$JULIA_CPU_TARGET" --check-bounds=yes \
        --threads=1 --project=run_hpc/1.13/test hpc/check_environment.jl 1.13
```

Repeat validation with `1.10` and its matching environment.
These focused checks run on the login node, complete suites and GPU execution require CI or allocations submitted by the user.

The [GitLab pipeline](../.gitlab-ci.yml) provides eight small validation jobs across both Julia versions.
Start it manually from the GitLab pipeline page after the commit reaches the mirror.
Pipelines run only from the GitLab UI, mirror pushes do not start jobs.
The Julia programs live in `hpc/ci/`, the YAML defines job resources and script invocations.
[environment.jl](ci/environment.jl) prepares and verifies artifacts, selects threading backends and precompiles packages in separate Julia processes.
[validate.jl](ci/validate.jl) runs the CPU, MPI and GPU checks, [mpi.jl](ci/mpi.jl) launches the MPI workers through the configured MPI installation.

| Check | Coverage |
|---|---|
| Preparation. | A fresh test environment, native library selection, parallel HDF5 and driverless CUDA precompilation are checked on the shell runner. |
| CPU. | The p4est and t8code advection items run serially and with four Polyester threads, the KernelAbstractions advection item runs with three threads. |
| MPI. | Three single threaded ranks write and read a collective HDF5 dataset and run the existing p4est and t8code MPI advection items. |
| GPU. | An existing CUDA item runs on one GPU, then two ranks verify distinct device UUIDs, exchange device buffers and run the existing p4est elixir with Float64 GPU storage. |

Preparation resolves dependencies once per pipeline and Julia version and publishes the project, manifest, threading preference and provenance.
Compute jobs restore the matching artifact and check its pipeline, commit, Julia version, CPU target, native installation marker and file hashes before loading Trixi.
The persistent depot supplies cached packages and compilation, compute jobs perform no dependency resolution.
Every selected item must be found, GPU checks require functional CUDA and MPI workers abort collectively on failures.
Serial RHS allocation checks use a 1,000 byte budget, threaded checks allow 5,000 bytes for thread scheduling overhead.

The pipeline defaults to the configured Otus depot, account and partitions, project variables can override `CI_DEPOT_ROOT`, `HPC_ACCOUNT`, `CPU_PARTITION` and `GPU_PARTITION`.
Both `otus-shell` and `otus-slurm` runners must be enabled.
[Jacamar scheduler parameters](https://ecp-ci.gitlab.io/docs/ci-users/ci-batch.html#scheduler-parameters) specify explicit tasks, CPUs, memory and execution time for each compute job.
The initial limits are thirty minutes for CPU and MPI jobs and twenty minutes for the combined GPU job, cold compilation costs still require measurement.
The GitLab timeout is seven days to accommodate queue wait, the runner must permit that timeout.
Logs and prepared environments are retained for four weeks, including artifacts from failed jobs.
Job logs report Slurm submit and start times, MPI bindings and the CPUs and threads visible to each worker.
`JACAMAR_SCHEDULER_JOB_PREFIX` identifies Trixi, the check and Julia version in Slurm job names.
The full test suites, benchmark and GitHub orchestration follow this validation pipeline.
