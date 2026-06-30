# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

"""
    ECAVCoefficientCallback(semi; interval=0,
                                  output_directory="out",
                                  filename="ecav_coefficients_max.dat")

Append one line per activation with the maximum entropy-correction artificial viscosity (ECAV)
coefficient over elements to `joinpath(output_directory, filename)`.

The callback uses the same step cadence as [`AnalysisCallback`](@ref): every `interval`
**accepted** time steps and once more when the simulation finishes. If `interval == 0`,
the callback is disabled.

This callback only supports [`SemidiscretizationArtificialViscosity`](@ref). It reads
`semi.cache.artificial_viscosity.coefficients` (per-element values updated in `rhs!` for the
hyperbolic part) and logs `maximum(coefficients)`. It is intended for **serial** runs;
MPI-parallel execution is not supported.

Each data line contains: accepted step count, simulation time `t`, time step `dt`,
and `ecav_coeff_max`. If there are no elements, `ecav_coeff_max` is recorded as `NaN`.

# Examples

```julia
ecav_cb = ECAVCoefficientCallback(semi, interval = 100)
callbacks = CallbackSet(summary_callback, analysis_callback, ecav_cb)
```
"""
mutable struct ECAVCoefficientCallback
    const interval::Int
    const output_directory::String
    const filename::String
end

function Base.show(io::IO, cb::DiscreteCallback{<:Any, <:ECAVCoefficientCallback})
    @nospecialize cb # reduce precompilation time
    ecc = cb.affect!
    print(io, "ECAVCoefficientCallback(interval=", ecc.interval, ")")
    return nothing
end

function Base.show(io::IO, ::MIME"text/plain",
                   cb::DiscreteCallback{<:Any, <:ECAVCoefficientCallback})
    @nospecialize cb # reduce precompilation time

    if get(io, :compact, false)
        show(io, cb)
    else
        ecc = cb.affect!
        setup = Pair{String, Any}["interval" => ecc.interval,
                                  "output directory" => abspath(normpath(ecc.output_directory)),
                                  "filename" => ecc.filename]
        summary_box(io, "ECAVCoefficientCallback", setup)
    end
end

function ECAVCoefficientCallback(semi::SemidiscretizationArtificialViscosity;
                               interval = 0,
                               output_directory = "out",
                               filename = "ecav_coefficients_max.dat")
    mpi_isparallel() &&
        error("ECAVCoefficientCallback does not support MPI-parallel runs")

    if !(interval isa Integer && interval >= 0)
        throw(ArgumentError("`interval` must be a non-negative integer (got interval = $interval)"))
    end

    condition = (u, t, integrator) -> interval > 0 &&
        (integrator.stats.naccept % interval == 0 || isfinished(integrator))

    affect! = ECAVCoefficientCallback(interval, output_directory, filename)
    return DiscreteCallback(condition, affect!;
                            save_positions = (false, false),
                            initialize = initialize!)
end

function initialize!(cb::DiscreteCallback{Condition, Affect!}, u_ode, t,
                     integrator) where {Condition, Affect! <: ECAVCoefficientCallback}
    ecc = cb.affect!
    @unpack output_directory, filename = ecc
    mkpath(output_directory)
    path = joinpath(output_directory, filename)
    open(path, "w") do io
        println(io, "#naccept time dt ecav_coeff_max svv_coeff_max norm_ecav norm_svv")
    end
    return nothing
end

function (ecc::ECAVCoefficientCallback)(integrator)
    semi = integrator.p
    ecav_coeffs = semi.cache.artificial_viscosity.coefficients
    svv_coeffs = semi.cache.artificial_viscosity.svv_coefficients
    ecav_coeff_max, norm_ecav = if isempty(ecav_coeffs)
        (float(eltype(coeffs))(NaN), float(eltype(coeffs))(NaN))
    else
        (maximum(ecav_coeffs), norm(ecav_coeffs))
    end
    svv_coeff_max, norm_svv = if isempty(svv_coeffs)
        (float(eltype(svv_coeffs))(NaN), float(eltype(svv_coeffs))(NaN))
    else
        (maximum(svv_coeffs), norm(svv_coeffs))
    end

    @unpack output_directory, filename = ecc
    path = joinpath(output_directory, filename)
    open(path, "a") do io
        println(io, integrator.stats.naccept, " ", integrator.t, " ", integrator.dt, " ",
                ecav_coeff_max, " ", svv_coeff_max, " ", norm_ecav, " ", norm_svv)
    end
    return nothing
end

end # @muladd
