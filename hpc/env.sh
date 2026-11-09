#!/usr/bin/env bash

# Source this file from Bash with the Julia minor version as its argument.
otus_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
otus_julia_minor=${1:?Expected Julia minor version.}
# The shared Otus depot holds packages, compilation caches and the native libraries.
export CI_DEPOT_ROOT=${CI_DEPOT_ROOT:-/scratch/hpc-prf-cobentx/depot}
case "$CI_DEPOT_ROOT" in
    /*) CI_DEPOT_ROOT=$(realpath -m -- "$CI_DEPOT_ROOT") ;;
    *) echo 'CI_DEPOT_ROOT must be an absolute path.' >&2; return 1 ;;
esac
case "$otus_julia_minor" in
    1.10)
        otus_julia_module=lang/JuliaHPC/1.10.12-foss-2025a
        otus_cpu_target=znver3,+avx512f,+avx512dq,+avx512cd,+avx512bw,+avx512vl
        ;;
    1.13)
        otus_julia_module=lang/JuliaHPC/1.13.3-foss-2025a
        otus_cpu_target=znver5
        ;;
    *) echo 'Expected Julia version 1.10 or 1.13 on Otus.' >&2; return 1 ;;
esac

if [[ -n ${SLURM_JOB_ID:-} ]]; then
    export TMPDIR=${TMPDIR:-/tmp}
else
    export TMPDIR="$otus_root/run_hpc/$otus_julia_minor/tmp"
fi
export TMP="$TMPDIR" TEMP="$TMPDIR"
mkdir -p -- "$TMPDIR" || return
otus_julia_threads=${JULIA_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-8}}
otus_mpi_cuda_support=${OMPI_MCA_opal_cuda_support:-0}
module purge || return
module load "$otus_julia_module" \
    lib/UCX-CUDA/1.18.0-GCCcore-14.2.0-CUDA-12.8.0 \
    data/HDF5/1.14.6-gompi-2025a || return

export JULIA_DEPOT_PATH="$CI_DEPOT_ROOT/$otus_julia_minor/julia-depot:$EBROOTJULIAHPC/local/share/julia:$EBROOTJULIAHPC/share/julia"
export JULIA_LOAD_PATH="@:$otus_root/hpc:@stdlib:$EBROOTJULIAHPC"
export JULIA_CPU_TARGET="$otus_cpu_target" JULIA_NUM_THREADS="$otus_julia_threads"
export JULIA_PKG_PRECOMPILE_AUTO=0
export JULIA_PKG_GC_AUTO=false
export JULIA_NUM_PRECOMPILE_TASKS=${JULIA_NUM_PRECOMPILE_TASKS:-8}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export CMAKE_BUILD_PARALLEL_LEVEL=${CMAKE_BUILD_PARALLEL_LEVEL:-4}
export OPAL_PREFIX="$EBROOTOPENMPI"
export OMPI_MCA_opal_cuda_support="$otus_mpi_cuda_support"
export OMPI_MCA_pml=ucx
# UCX reads the cache setting before MPI.jl initializes.
export UCX_MEMTYPE_CACHE=no
export UCX_ERROR_SIGNALS=${UCX_ERROR_SIGNALS-}
export JULIA_CUDA_MEMORY_POOL=${JULIA_CUDA_MEMORY_POOL:-none}
