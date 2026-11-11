# HPC setup

The setup on Otus currently supports Julia 1.10 and 1.13 with system OpenMPI, parallel HDF5 and local CUDA 12.8.
[env.sh](env.sh) and the [preferences project](Project.toml) define the used toolchain and must be updated together.
See the [system MPI documentation](../docs/src/parallelization.md) for library requirements.

## Running pipelines

The [GitLab pipeline](../.gitlab-ci.yml) runs validation checks and scaling benchmarks on Otus.
It starts from the [GitHub workflow](../.github/workflows/hpc.yml), a PR command or the GitLab UI.

Start a run on GitHub under Actions with `Run workflow` of the `HPC` workflow.
The inputs take `all` or the listed values joined by `+`, `checks` and `benchmarks` also take `none`.

| Input | Values | Default | GitLab variable |
|---|---|---|---|
| `checks` | `cpu`, `mpi`, `gpu` | `all` | `HPC_CHECKS` |
| `benchmarks` | `threads`, `mpi`, `gpu` | `none` | `HPC_BENCHMARKS` |
| `julia` | `1.10`, `1.13` | `all` | `HPC_JULIA` |

Maintainers can run reviewed PR commits with a comment on the PR.

```text
/hpc <commit>
/hpc <commit> <commit> checks=gpu julia=1.13
/hpc <commit> checks=none benchmarks=all
```

Commits are named by their full 40 character hash and must belong to the PR.

## Preparation

A prepare job per selected Julia version builds the environment that all compute jobs use.

| File | Role |
|---|---|
| [env.sh](env.sh) | Loads the modules and sets the Julia depot, load path, CPU target and the MPI settings. |
| [install_libraries.sh](install_libraries.sh) | Builds p4est and t8code from the pinned [CMake recipe](libraries/CMakeLists.txt) into the shared depot. |
| [prepare.jl](ci/prepare.jl) | Resolves the test environment under `run_hpc/<version>/test`. |
| [check_environment.jl](check_environment.jl) | Verifies preferences, loaded native libraries, parallel HDF5 and driverless CUDA precompilation. |

The job publishes the environment files as an artifact, the depots are shared at a fixed location.

Compute jobs then restore the artifacts and are working with an offline depot.

## Validation

The setup provides three short validation jobs.

| Check | Coverage |
|---|---|
| CPU | Run p4est and t8code advection serially and with four Polyester threads, using the items of [test_threaded.jl](../test/test_threaded.jl), run KernelAbstractions advection with three threads. |
| MPI | Run one rank per node on two nodes, collective HDF5 I/O and the existing p4est and t8code MPI advection items. |
| GPU | Run the existing CUDA item, then verify distinct GPUs, device buffer exchange and a distributed p4est solve with Float64 GPU storage. |

## Benchmarks

[benchmark/scaling](../benchmark/scaling) holds the scaling benchmarks.
Its `run.jl` measures the [performance index (PID)](../docs/src/performance.md) of a case, a file that names an existing elixir and its `trixi_include` overrides.
Each scaling point runs as its own job, `combine.jl` then writes `benchmark.json` with the PIDs and `efficiency.json` with the parallel efficiency relative to the baseline.
Benchmarks run with `--check-bounds=auto` instead of the `yes` of the checks.

| Series | Points | Baseline | Allocation |
|---|---|---|---|
| Threads | [euler_source_terms.jl](../benchmark/scaling/euler_source_terms.jl) with about 2.1 million degrees of freedom on 1, 8, 24, 48 and 96 threads. | 8 threads | A whole node, the threads fill the cores of the first socket in order. |
| MPI | The same case on 1, 8, 24 and 96 ranks of one node and 192 ranks on two nodes. | 8 ranks | Whole nodes, the ranks fill the cores of the first socket of each node in order. |
| GPU | [taylor_green_vortex_cuda.jl](../benchmark/scaling/taylor_green_vortex_cuda.jl) on one and two H100 GPUs. | 1 GPU | Two GPUs with 24 cores each. |

Because the benchmarks are memory bound, the baseline is one CCD of eight cores with one L3 cache domain.

The runs are done on exclusive nodes and the threads/ranks are pinned densely into the CCDs and onto the firstsocket.

The number of time steps is proportional to the count, to get a similar runtime for every core count.

The fastest of three iterations is the measured result.

The published metrics are the PID for every core count and an efficiency based on the 8-core run.

The history of the default branch is recorded on the `hpc-results` branch and the dashboard is deployed to GitHub Pages under `/bench/`.

The workflow defines alert and failure thresholds per suite, PR runs compare with the history without recording.

## Setup

Follow the [setup of `pc2/hpc-cb-workflow`](https://github.com/pc2/hpc-cb-workflow#setup) for GitLab mirroring, tokens and GitHub secrets.

The mirror is the GitLab project `pc2-ci/julia/trixi.jl` on `git.uni-paderborn.de` with `otus-shell` and `otus-slurm` runners.

The following variables are needed additionally.

| Variable | Where | Current value |
|---|---|---|
| `GITLAB_PROJECT_ID` | GitHub Actions repository variable | `10376` |
| `HPC_ACCOUNT` | GitLab CI/CD variable | `hpc-prf-cobentx` |
| `CPU_PARTITION` | GitLab CI/CD variable | `normal` |
| `GPU_PARTITION` | GitLab CI/CD variable | `gpu_h100` |
| `CI_DEPOT_ROOT` | GitLab CI/CD variable, optional | Default of [env.sh](env.sh) |
