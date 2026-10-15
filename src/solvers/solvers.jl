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
        set_zero_element!(du, element)
    end

    return nothing
end

@inline function set_zero_element!(du, element)
    du[.., element] .= zero(eltype(du))
    return nothing
end

# Workaround for https://github.com/JuliaArrays/FixedSizeArrays.jl/issues/200:
# On Julia v1.13 and newer, `fill!` of (views of) `FixedSizeArray`s does not vectorize
# since the bounds checks cannot be elided. Thus, we set the contiguous memory of the
# `element` in the parent `Vector` directly.
@inline function set_zero_element!(du::Union{VectorBackedArray,
                                             StaticSizeArray{<:Any, <:Any, <:Any,
                                                             <:Vector}},
                                   element)
    n = length(du) ÷ size(du, ndims(du))
    data = parent(du)
    @boundscheck checkbounds(data, (element - 1) * n + 1)
    @boundscheck checkbounds(data, element * n)
    @inbounds for i in ((element - 1) * n + 1):(element * n)
        data[i] = zero(eltype(du))
    end
    return nothing
end

function set_zero!(::Backend, du, dg, cache)
    # Broadcasting is parallel on the GPU
    du .= zero(eltype(du))
    return nothing
end

# Abstract supertype of the GPU kernels for `VolumeIntegralFluxDifferencing`, which can
# be selected via `Tuning(; flux_differencing_kernel)`
abstract type AbstractFluxDifferencingKernel end

"""
    HalfSweep()

Selects the "half sweep" GPU kernel for [`VolumeIntegralFluxDifferencing`](@ref),
see [`Tuning`](@ref) and [`trixi_adapt`](@ref). This is the default on GPUs. The
KernelAbstractions.jl CPU backend does not support this kernel and uses
[`FullSweep`](@ref) instead.

All diagonal entries of `derivative_split` are zero. Thus, we can skip the
computation of the diagonal terms. In addition, we use the symmetry of the
`volume_flux` to save half of the possible two-point flux computations. Each flux
is staged in shared memory, so that both nodes of the pair can use it. The half
sweep is distributed cyclically over the threads, so that each of them evaluates
the same number of two-point fluxes.

On (recent) NVIDIA GPUs, this kernel should be faster than [`FullSweep`](@ref)
and [`FullSweepGlobal`](@ref) for most configurations. [`FullSweep`](@ref) can
be competitive for systems with few variables and high polynomial degrees, and
for older-generation GPUs such as an AMD MI210.

However, both [`HalfSweep`](@ref) and [`FullSweep`](@ref) use some local storage.
For systems with many variables and/or high polynomial degrees, this can exceed
the available storage on current GPUs, and the resulting workgroup sizes may
exceed limits on some systems. For this case, [`FullSweepGlobal`](@ref)
provides a fallback option that is typically slower but should work for all
configurations.

See also [`FullSweep`](@ref) and [`FullSweepGlobal`](@ref).

For details on the cyclic distribution see Section 4.1 (Eq. 6) of
- Waterhouse, Waruszewski, Wilcox, Giraldo (2026)
  GPU Performance of an Entropy-Stable Discontinuous Galerkin Euler Solver
  with Non-Conservative Terms
  [arXiv: 2605.16684](https://arxiv.org/abs/2605.16684)
"""
struct HalfSweep <: AbstractFluxDifferencingKernel end

"""
    FullSweep()

Selects the "full sweep" GPU kernel for [`VolumeIntegralFluxDifferencing`](@ref),
see [`Tuning`](@ref) and [`trixi_adapt`](@ref).

All two-point fluxes are computed for each node.
This doubles the number of flux evaluations,
but no flux has to be exchanged between the threads.
Shared memory is used to avoid repeated slow reads
from the solution vector `u`.

See [`HalfSweep`](@ref) for guidance on choosing between the kernels.
"""
struct FullSweep <: AbstractFluxDifferencingKernel end

"""
    FullSweepGlobal()

Selects the "global full sweep" GPU kernel for [`VolumeIntegralFluxDifferencing`](@ref),
see [`Tuning`](@ref) and [`trixi_adapt`](@ref).

Same sweep as [`FullSweep`](@ref), but the solution is read from global memory
instead of being copied in shared memory. Therefore, the kernel does not need
synchronization barriers and thus it is not limited by the number of variables
and the number of nodes in each element. For that reason it is useful
in case the faster kernels [`FullSweep`](@ref) and [`HalfSweep`](@ref)
are limited by the shared memory, due to either high number of variables, high
polynomial degree, or both.
"""
struct FullSweepGlobal <: AbstractFluxDifferencingKernel end

# Return the flux differencing kernel used on the `backend` for the setting `kernel` of a
# `Tuning`. `nothing` selects the default of the `backend`.
@inline flux_differencing_kernel(::Backend, ::Nothing) = HalfSweep()
@inline flux_differencing_kernel(::KernelAbstractions.CPU, ::Nothing) = FullSweep()
# Fallback for CPU KernelAbstractions backend
# This fallback is necessary to run tests on KernelAbstractions CPU backend
@inline flux_differencing_kernel(::KernelAbstractions.CPU, ::HalfSweep) = FullSweep()
@inline flux_differencing_kernel(::Backend, kernel) = kernel

"""
    Tuning(; flux_differencing_kernel = nothing)

Performance settings of the GPU and KernelAbstractions.jl kernels of a semidiscretization
on a [`P4estMesh`](@ref) or [`T8codeMesh`](@ref). They select between different
implementations of the same discretization, which are equivalent in exact arithmetic.
Thus, they do not change the numerical method, but they can change the performance, the
resources required on a GPU (e.g., shared memory), and the results at the level of
floating-point round-off errors (e.g., due to a different order of operations).
Pass them to [`trixi_adapt`](@ref), e.g.,
```julia
semi = trixi_adapt(semi; storage_type = CuArray, real_type = Float32,
                   tuning = Tuning(flux_differencing_kernel = FullSweep()))
```

Settings that are `nothing` keep their current value in the semidiscretization. A new
semidiscretization uses `Tuning()`, i.e., all settings are `nothing`, which selects the
defaults of the backend (e.g., a GPU or the KernelAbstractions.jl CPU backend) given
below. Thus, the defaults follow the backend if the semidiscretization is adapted to
another `storage_type`.

Optional keyword arguments:
- `flux_differencing_kernel`: GPU kernel used for [`VolumeIntegralFluxDifferencing`](@ref),
  one of [`HalfSweep()`](@ref) (default on GPUs), [`FullSweep()`](@ref) (default on the
  KernelAbstractions.jl CPU backend, which does not support [`HalfSweep()`](@ref)), or
  [`FullSweepGlobal()`](@ref).
"""
struct Tuning{FluxDifferencingKernel <:
              Union{Nothing, AbstractFluxDifferencingKernel}}
    # The type parameters restrict the settings to the accepted values (or `nothing`),
    # so that invalid settings throw an error when creating the `Tuning` instead of
    # when launching a kernel.
    flux_differencing_kernel::FluxDifferencingKernel
end

function Tuning(; flux_differencing_kernel = nothing)
    return Tuning(flux_differencing_kernel)
end

# The tuning stored in the `cache` of a new semidiscretization. All settings are
# `nothing`, so that the defaults of the backend are used, see `resolve_tuning`.
default_tuning() = Tuning()

# Return the settings of `tuning` used on the `backend`, where settings that are
# `nothing` are replaced by the defaults of the `backend`.
function resolve_tuning(backend, tuning::Tuning)
    kernel = flux_differencing_kernel(backend, tuning.flux_differencing_kernel)
    return Tuning(; flux_differencing_kernel = kernel)
end

# Return the settings of `new_tuning` that are not `nothing` and the settings of
# `tuning` otherwise.
function merge_tuning(tuning::Tuning, new_tuning::Tuning)
    settings = map(fieldnames(Tuning)) do name
        setting = getfield(new_tuning, name)
        return setting === nothing ? getfield(tuning, name) : setting
    end
    return Tuning(settings...)
end

# define types for parabolic solvers
include("solvers_parabolic.jl")

include("dg.jl")
include("dg_gpu.jl")
include("dgmulti/dgmulti.jl")
end # @muladd
