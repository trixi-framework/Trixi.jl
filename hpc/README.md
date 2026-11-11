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

The setup provides three small validation jobs for each Julia version.

| Check | Coverage |
|---|---|
| CPU | Run p4est and t8code advection serially and with four Polyester threads, using the items of [test_threaded.jl](../test/test_threaded.jl), run KernelAbstractions advection with three threads. |
| MPI | Run one rank per node on two nodes, collective HDF5 I/O and the existing p4est and t8code MPI advection items. |
| GPU | Run the existing CUDA item, then verify distinct GPUs, device buffer exchange and a distributed p4est solve with Float64 GPU storage. |

Every selected item must be found, GPU checks require functional CUDA and MPI workers abort collectively on failures.

## Benchmarks

[run.jl](../benchmark/run.jl) measures the performance index (PID) of an existing elixir, the pipeline variables define the elixir and its `trixi_include` overrides.
The PID is the wall time per right-hand side evaluation and global degree of freedom, multiplied by the number of ranks and threads, and stays constant under ideal scaling.
Each scaling point runs as its own job, [scaling.jl](ci/scaling.jl) then writes `benchmark.json` with the PIDs and `efficiency.json` with the parallel efficiency relative to the smallest count.
Benchmarks run with `--check-bounds=auto` instead of the `yes` of the checks.

| Series | Points | Allocation |
|---|---|---|
| Threads | `p4est_3d_dgsem/elixir_euler_source_terms.jl` with about 2.1 million degrees of freedom on 1, 8, 24, 48 and 96 threads. | A whole node, the threads fill the cores of the first socket in order. |
| MPI | The same elixir on 1, 24 and 96 ranks of one node and 192 ranks on two nodes. | Whole nodes, the ranks fill the cores of the first socket of each node in order. |
| GPU | The Taylor-Green vortex of [benchmark/CUDA](../benchmark/CUDA) on one and two H100 GPUs. | Two GPUs with 24 cores each. |

The CPU case is limited by memory bandwidth, so CPU points reserve whole nodes and pin threads and ranks to cores with [ThreadPinning.jl](https://github.com/carstenbauer/ThreadPinning.jl) and OpenMPI.
The thread series stays within one socket, since Trixi.jl initializes its arrays on the main thread and the pages reside in its NUMA domain.

The number of time steps is proportional to the count, so every CPU point measures about ten seconds under ideal scaling.
`run.jl` keeps the fastest of three repetitions and fails when the elixir stops before its last step.

The [GitHub workflow](../.github/workflows/hpc.yml) publishes each series as a PID suite and an efficiency suite per Julia version.
Runs record the history on the `hpc-results` branch and deploy its dashboard with GitHub Pages under `/bench/`.
The workflow defines alert and failure thresholds per suite, PR runs compare with the history without recording.

## Slurm resources and logs

Slurm jobs use the [express QoS](https://upb-pc2.atlassian.net/wiki/spaces/PC2DOK/pages/1902070) with limits of thirty minutes for CPU, MPI and benchmark jobs and twenty minutes for GPU checks.
The GitLab timeout is seven days to accommodate queue wait.
Logs and prepared environments are retained for four weeks, including artifacts from failed jobs.

Job logs report Slurm submit and start times, MPI bindings and the node, CPUs and threads visible to each worker, benchmark jobs add the NUMA layout and thread placement.
`JACAMAR_SCHEDULER_JOB_PREFIX` identifies Trixi, the check and Julia version in Slurm job names.

## Pipeline setup

The [GitHub workflow](../.github/workflows/hpc.yml) and the [command workflow](../.github/workflows/hpc-command.yml) pin `pc2/hpc-cb-workflow` to the same reviewed commit, update both together.
Follow the [workflow setup](https://github.com/pc2/hpc-cb-workflow#setup) for GitLab mirroring, tokens and GitHub secrets.
Use GitLab project `pc2-ci/julia/trixi.jl` on `git.uni-paderborn.de` and set the Actions repository variable `GITLAB_PROJECT_ID` to `10376`.

Both `otus-shell` and `otus-slurm` runners must be enabled and permit the job timeout of seven days.

Jacamar submits the Slurm jobs before they source [env.sh](env.sh), so the GitLab project defines the Slurm account and partitions as CI/CD variables.
Preparation fails when one of them is missing.
A project variable `CI_DEPOT_ROOT` overrides the default depot.

```bash
project=https://git.uni-paderborn.de/pc2-ci/julia/trixi.jl
glab variable set HPC_ACCOUNT hpc-prf-cobentx -R "$project"
glab variable set CPU_PARTITION normal -R "$project"
glab variable set GPU_PARTITION gpu_h100 -R "$project"
```
