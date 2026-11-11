# HPC setup

The Otus setup supports Julia 1.10 and 1.13 with system OpenMPI, parallel HDF5 and local CUDA 12.8.
[env.sh](env.sh) and the [preferences project](Project.toml) define one fixed toolchain and must be updated together.
See the [system MPI documentation](../docs/src/parallelization.md) for library requirements.

## Environment

`env.sh` defaults `CI_DEPOT_ROOT` to the shared Otus depot, set it to another absolute directory for a separate depot.
Load the environment and install the pinned native libraries from Bash.

```bash
source hpc/env.sh 1.13 &&
    bash hpc/install_libraries.sh
```

`env.sh` loads the modules and sets Julia depot and load paths that exclude the user depot and expose cluster preferences before resolution and package loading.
Pass `-C "$JULIA_CPU_TARGET"` to Julia so runtime compilation matches package compilation.

OpenMPI selects UCX for CPU and GPU communication.
`UCX_MEMTYPE_CACHE=no` disables cached pointer classification and must be set before Julia starts, following the [MPI.jl compatibility guidance](https://juliaparallel.org/MPI.jl/stable/knownissues/#Memory-cache).

Login node temporary files stay in the checkout, compute jobs keep the default temporary directory.
The job's `/tmp` resides on GPFS, so compute jobs place the OpenMPI session directory in `/dev/shm`.

`CMAKE_BUILD_PARALLEL_LEVEL`, `JULIA_NUM_PRECOMPILE_TASKS`, `JULIA_NUM_THREADS` and `OPENBLAS_NUM_THREADS` control concurrency.

## Native libraries

[install_libraries.sh](install_libraries.sh) builds the pinned [CMake recipe](libraries/CMakeLists.txt) under a shared lock and publishes its marker after checking the installed libraries.
The separate p4est 2.8 build preserves library versions distinct from t8code's dependencies.
CMake is invoked directly because loading its module would override Julia's libcurl.

Both Julia versions share `/scratch/hpc-prf-cobentx/depot/native/otus-foss-2025a-v2`.
The installation fingerprint covers the native recipe and OpenMPI, zlib and CMake paths, incompatible installations are rejected.
Native updates require a new versioned path in the installer and preferences.
Use `bash hpc/install_libraries.sh --print-fingerprint` to record it without modifying the installation.

## Environment check

[check_environment.jl](check_environment.jl) verifies effective preferences, loaded native libraries, parallel HDF5 and driverless CUDA precompilation.
It runs the existing basic p4est and t8code advection items, which generate their own meshes.

Use a test environment that develops this checkout and contains `test/Project.toml` dependencies, for example `run_hpc/1.13/test`.
`run_hpc/` holds the prepared environments and login node temporary files per Julia version, the existing `run*/` pattern of `.gitignore` keeps it out of git.
[environment.jl](ci/environment.jl) prepares it as in CI, it reads the checkout, Julia version and commit from CI variables and holds the lock of the shared depot.

```bash
source hpc/env.sh 1.13 &&
    export CI_PROJECT_DIR=$PWD JULIA_VERSION=1.13 CI_COMMIT_SHA=$(git rev-parse HEAD) &&
    flock "$CI_DEPOT_ROOT/1.13/prepare.lock" \
        julia --startup-file=no -C "$JULIA_CPU_TARGET" --check-bounds=yes \
        --project=hpc hpc/ci/environment.jl prepare
```

Then run the check.

```bash
source hpc/env.sh 1.13 &&
    julia --startup-file=no -C "$JULIA_CPU_TARGET" --check-bounds=yes \
        --threads=1 --project=run_hpc/1.13/test hpc/check_environment.jl 1.13
```

Repeat validation with `1.10` and its matching environment.
These focused checks run on the login node, complete suites and GPU execution require CI or allocations submitted by the user.

## Validation pipeline

The [GitLab pipeline](../.gitlab-ci.yml) provides eight small validation jobs across both Julia versions.
Pipelines start from the GitLab UI, the [GitHub workflow](../.github/workflows/hpc.yml) or a PR command, mirror pushes do not start jobs.
GitHub triggers require `EXPECTED_COMMIT_SHA` to match the mirrored commit, a stale or advanced mirror cannot run another revision.

| Check | Coverage |
|---|---|
| Preparation | Resolve the environment, verify native libraries and parallel HDF5, precompile both threading backends and CUDA without a driver. |
| CPU | Run p4est and t8code advection serially and with four Polyester threads, run KernelAbstractions advection with three threads. |
| MPI | Run one rank per node on two nodes, collective HDF5 I/O and the existing p4est and t8code MPI advection items. |
| GPU | Run the existing CUDA item, then verify distinct GPUs, device buffer exchange and a distributed p4est solve with Float64 GPU storage. |

The YAML defines resources and invokes [environment.jl](ci/environment.jl) for preparation and artifact verification, [validate.jl](ci/validate.jl) for checks and [mpi.jl](ci/mpi.jl) for MPI launching.
The threaded checks use two items for the basic p4est and t8code advection elixirs in [test_threaded.jl](../test/test_threaded.jl).

Preparation resolves dependencies once per pipeline and Julia version and publishes the project, manifest, threading preference and provenance.
The test environment gains the dependencies of the [preferences project](Project.toml), currently ThreadPinning.jl for the benchmarks.

Compute jobs restore the matching artifact and check its pipeline, commit, Julia version, CPU target, native installation marker and file hashes before loading Trixi.
The persistent depot caches packages and compilation, compute jobs perform no dependency resolution.

Every selected item must be found, GPU checks require functional CUDA and MPI workers abort collectively on failures.
Serial RHS allocation checks use a 1,000 byte budget, threaded checks allow 5,000 bytes for thread scheduling overhead.

## Slurm resources and logs

The Slurm account and partitions come from GitLab project variables, see the [setup](#setup).
Both `otus-shell` and `otus-slurm` runners must be enabled.

Slurm jobs use the [express QoS](https://upb-pc2.atlassian.net/wiki/spaces/PC2DOK/pages/1902070).
[Jacamar scheduler parameters](https://ecp-ci.gitlab.io/docs/ci-users/ci-batch.html#scheduler-parameters) specify explicit tasks, CPUs, memory and execution time for each compute job.
Slurm limits are thirty minutes for CPU, MPI and benchmark jobs and twenty minutes for GPU checks.
The GitLab timeout is seven days to accommodate queue wait, the runner must permit that timeout.

Logs and prepared environments are retained for four weeks, including artifacts from failed jobs.

Job logs report Slurm submit and start times, MPI bindings and the node, CPUs and threads visible to each worker.
CPU MPI jobs log the UCX transport selection, CPU binding checks compare ranks within each node.
`JACAMAR_SCHEDULER_JOB_PREFIX` identifies Trixi, the check and Julia version in Slurm job names.

## Benchmarks

`HPC_BENCHMARKS` selects scaling benchmarks independently of the checks, `HPC_CHECKS=none` runs benchmarks without validation.

[run.jl](../benchmark/run.jl) measures the performance index (PID) of an existing elixir, the pipeline variables define the elixir and its `trixi_include` overrides.
Each scaling point runs as its own matrix job, a final job per series runs [scaling.jl](ci/scaling.jl) on the shell runner and writes `benchmark.json` with the PIDs and `efficiency.json` with the parallel efficiency relative to the smallest count.

Benchmarks run with automatic bounds checking, preparation precompiles that mode when benchmarks are selected.

| Series | Points | Allocation |
|---|---|---|
| Threads | `p4est_3d_dgsem/elixir_euler_source_terms.jl` with about 2.1 million degrees of freedom on 1, 8, 24, 48 and 96 threads. | A whole node, the threads fill the cores of the first socket in order. |
| MPI | The same elixir on 1, 24 and 96 ranks of one node and 192 ranks on two nodes. | Whole nodes, the ranks fill the cores of the first socket of each node in order. |
| GPU | The Taylor-Green vortex of [benchmark/CUDA](../benchmark/CUDA) on one and two H100 GPUs. | Two GPUs with 24 cores each. |

The CPU case is limited by memory bandwidth, so the measurements depend on other jobs on the node and on the placement of threads and ranks.
CPU points therefore reserve whole nodes, [ThreadPinning.jl](https://github.com/carstenbauer/ThreadPinning.jl) pins the threads and OpenMPI binds the ranks to consecutive cores.
Compact placement fills one cache domain after the other, so the efficiency shows how the code saturates the memory bandwidth of a socket.
The thread series stays within one socket, since Trixi.jl initializes its arrays on the main thread and the pages reside in its NUMA domain.
The jobs log the CPU and NUMA layout, the thread placement and the rank bindings.

The steps grow with the count, so every CPU point measures about ten seconds under ideal scaling.
`run.jl` repeats the single step and the measured run three times and keeps the fastest repetition of each.
`run.jl` fails when the elixir stops before its last step, an aborted solve would report the PID of a wrong solution.

The [GitHub caller](../.github/workflows/hpc.yml) publishes each series as a PID suite and an efficiency suite per Julia version.
Runs record the history on the `hpc-results` branch and deploy its dashboard with GitHub Pages under `/bench/`.
The caller defines alert and failure thresholds per suite, PR runs compare with the history without recording.

## Setup

The [caller](../.github/workflows/hpc.yml) and the [command workflow](../.github/workflows/hpc-command.yml) pin `pc2/hpc-cb-workflow` to the same reviewed commit, update both together.
Follow the [workflow setup](https://github.com/pc2/hpc-cb-workflow#setup) for GitLab mirroring, tokens and GitHub secrets.
Use GitLab project `pc2-ci/julia/trixi.jl` on `git.uni-paderborn.de` and set the Actions repository variable `GITLAB_PROJECT_ID` to `10376`.

Jacamar submits the Slurm jobs before they source [env.sh](env.sh), so the GitLab project defines the Slurm settings as CI/CD variables.
Preparation fails when one of them is missing.
A project variable `CI_DEPOT_ROOT` overrides the default depot.

```bash
project=https://git.uni-paderborn.de/pc2-ci/julia/trixi.jl
glab variable set HPC_ACCOUNT hpc-prf-cobentx -R "$project"
glab variable set CPU_PARTITION normal -R "$project"
glab variable set GPU_PARTITION gpu_h100 -R "$project"
```

Start a run from GitHub Actions, HPC, Run workflow on `main`.

`checks` selects `all`, `none` or `cpu`, `mpi` and `gpu` joined by `+` and defaults to `all`.
`benchmarks` selects `all`, `none` or `threads`, `mpi` and `gpu` joined by `+` and defaults to `none`.
`julia` selects `all` or `1.10` and `1.13` joined by `+` and defaults to `all`.
The workflow checks the values against the caller's patterns and stops before anything reaches GitLab.

Preparation runs only for the selected Julia versions and only when checks or benchmarks are selected.

Monitoring continues automatically after long queue waits, see [monitoring and results](https://github.com/pc2/hpc-cb-workflow#how-it-runs).

Maintainers run reviewed PR commits with a comment on the PR, the command workflow dispatches the caller once per commit.

```text
/hpc <commit>
/hpc <commit> <commit> checks=gpu julia=1.13
/hpc <commit> checks=none benchmarks=all
```

Commits are named by their full 40 character hash and must belong to the PR, later pushes need another comment.
Commands and manual runs of PR commits or other branches need repository write access.
Options apply to all named commits and use the names of the workflow inputs.
See the [PR command](https://github.com/pc2/hpc-cb-workflow#pr-command) for authorization and replies.

Pipelines started from the GitLab UI run all checks and no benchmarks unless `HPC_CHECKS`, `HPC_BENCHMARKS` or `HPC_JULIA` are set as pipeline variables.
