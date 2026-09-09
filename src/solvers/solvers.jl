# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

function set_zero!(du, dg, cache)
    set_zero!(trixi_backend(du), du, dg, cache)

    return nothing
end

# Used by both `dg::DGSEM` and `dg::FDSBP`
function set_zero!(::Nothing, du, dg, cache)
    # du .= zero(eltype(du)) doesn't scale when using multiple threads.
    # See https://github.com/trixi-framework/Trixi.jl/pull/924 for a performance comparison.
    @threaded for element in eachelement(dg, cache)
        du[.., element] .= zero(eltype(du))
    end

    return nothing
end

function set_zero!(::Backend, du, dg, cache)
    # Broadcasting is parallel on the GPU
    du .= zero(eltype(du))
    return nothing
end

"""
    HalfSweep()

Selects the "half sweep" GPU kernel for [`VolumeIntegralFluxDifferencing`](@ref),
see [`semidiscretize`](@ref). This is the default.

All diagonal entries of `derivative_split` are zero. Thus, we can skip the
computation of the diagonal terms. In addition, we use the symmetry of the
`volume_flux` to save half of the possible two-point flux computations. Each flux
is staged in shared memory, so that both nodes of the pair can use it. The half
sweep is distributed cyclically over the threads, so that each of them evaluates
the same number of two-point fluxes.

On NVIDIA GPUs, this kernel should be faster than [`FullSweep`](@ref) for most
configurations. [`FullSweep`](@ref) can be competitive for systems with few
variables and high polynomial degrees.

See also [`FullSweep`](@ref) and [`FullSweepGlobal`](@ref).

For details on the cyclic distribution see Section 4.1 (Eq. 6) of
- Waterhouse, Waruszewski, Wilcox, Giraldo (2026)
  GPU Performance of an Entropy-Stable Discontinuous Galerkin Euler Solver
  with Non-Conservative Terms
  [arXiv: 2605.16684](https://arxiv.org/abs/2605.16684)
"""
struct HalfSweep end

"""
    FullSweep()

Selects the "full sweep" GPU kernel for [`VolumeIntegralFluxDifferencing`](@ref),
see [`semidiscretize`](@ref).

All two-point fluxes are computed for each node.
This doubles the number of flux evaluations,
but no flux has to be exchanged between the threads.
Shared memory is used to avoid repeated slow reads
from the solution vector `u`.

See [`HalfSweep`](@ref) for guidance on choosing between the kernels.
"""
struct FullSweep end

"""
    FullSweepGlobal()

Selects the "global full sweep" GPU kernel for [`VolumeIntegralFluxDifferencing`](@ref),
see [`semidiscretize`](@ref).

Same sweep as [`FullSweep`](@ref), but the solution is read from global memory
instead of being copied in shared memory. Therefore, the kernel does not need
synchronization barriers and thus it is not limited by the number of variables
and the number of nodes in each element. For that reason it is useful
in case the faster kernels [`FullSweep`](@ref) and [`HalfSweep`](@ref)
are limited by the shared memory, due to either high number of variables, high
polynomial degree, or both.
"""
struct FullSweepGlobal end

# Fallback for CPU KernelAbstractions backend
# This fallback is necessary to run Kernel
@inline flux_differencing_kernel(::KernelAbstractions.CPU, ::HalfSweep) = FullSweep()
@inline flux_differencing_kernel(::Backend, kernel) = kernel

# define types for parabolic solvers
include("solvers_parabolic.jl")

include("dg.jl")
include("dg_gpu.jl")
include("dgmulti/dgmulti.jl")
end # @muladd
