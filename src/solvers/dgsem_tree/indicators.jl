# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

abstract type AbstractIndicator end

function create_cache(typ::Type{IndicatorType},
                      semi) where {IndicatorType <: AbstractIndicator}
    create_cache(typ, mesh_equations_solver_cache(semi)...)
end

function get_element_variables!(element_variables, indicator::AbstractIndicator,
                                ::VolumeIntegralShockCapturingHG)
    element_variables[:indicator_shock_capturing] = indicator.cache.alpha
    return nothing
end

"""
    IndicatorHennemannGassner(equations::AbstractEquations, basis;
                              alpha_max=0.5,
                              alpha_min=0.001,
                              alpha_smooth=true,
                              variable)
    IndicatorHennemannGassner(semi::AbstractSemidiscretization;
                              alpha_max=0.5,
                              alpha_min=0.001,
                              alpha_smooth=true,
                              variable)

Indicator used for shock-capturing (when passing the `equations` and the `basis`)
or adaptive mesh refinement (AMR, when passing the `semi`).

See also [`VolumeIntegralShockCapturingHG`](@ref).

## References

- Hennemann, Gassner (2020)
  "A provably entropy stable subcell shock capturing approach for high order split form DG"
  [arXiv: 2008.12044](https://arxiv.org/abs/2008.12044)
"""
struct IndicatorHennemannGassner{RealT <: Real, Variable, Cache} <: AbstractIndicator
    alpha_max::RealT
    alpha_min::RealT
    alpha_smooth::Bool
    variable::Variable
    cache::Cache
end

# this method is used when the indicator is constructed as for shock-capturing volume integrals
function IndicatorHennemannGassner(equations::AbstractEquations, basis;
                                   alpha_max = 0.5,
                                   alpha_min = 0.001,
                                   alpha_smooth = true,
                                   variable)
    alpha_max, alpha_min = promote(alpha_max, alpha_min)
    cache = create_cache(IndicatorHennemannGassner, equations, basis)
    IndicatorHennemannGassner{typeof(alpha_max), typeof(variable), typeof(cache)}(alpha_max,
                                                                                  alpha_min,
                                                                                  alpha_smooth,
                                                                                  variable,
                                                                                  cache)
end

# this method is used when the indicator is constructed as for AMR
function IndicatorHennemannGassner(semi::AbstractSemidiscretization;
                                   alpha_max = 0.5,
                                   alpha_min = 0.001,
                                   alpha_smooth = true,
                                   variable)
    alpha_max, alpha_min = promote(alpha_max, alpha_min)
    cache = create_cache(IndicatorHennemannGassner, semi)
    IndicatorHennemannGassner{typeof(alpha_max), typeof(variable), typeof(cache)}(alpha_max,
                                                                                  alpha_min,
                                                                                  alpha_smooth,
                                                                                  variable,
                                                                                  cache)
end

function Base.show(io::IO, indicator::IndicatorHennemannGassner)
    @nospecialize indicator # reduce precompilation time

    print(io, "IndicatorHennemannGassner(")
    print(io, indicator.variable)
    print(io, ", alpha_max=", indicator.alpha_max)
    print(io, ", alpha_min=", indicator.alpha_min)
    print(io, ", alpha_smooth=", indicator.alpha_smooth)
    print(io, ")")
end

function Base.show(io::IO, ::MIME"text/plain", indicator::IndicatorHennemannGassner)
    @nospecialize indicator # reduce precompilation time
    setup = [
        "indicator variable" => indicator.variable,
        "max. α" => indicator.alpha_max,
        "min. α" => indicator.alpha_min,
        "smooth α" => (indicator.alpha_smooth ? "yes" : "no")
    ]
    summary_box(io, "IndicatorHennemannGassner", setup)
end

function (indicator_hg::IndicatorHennemannGassner)(u, mesh, equations, dg::DGSEM, cache;
                                                   kwargs...)
    @unpack alpha_smooth = indicator_hg
    @unpack alpha, alpha_tmp = indicator_hg.cache
    # TODO: Taal refactor, when to `resize!` stuff changed possibly by AMR?
    #       Shall we implement `resize!(semi::AbstractSemidiscretization, new_size)`
    #       or just `resize!` whenever we call the relevant methods as we do now?
    resize!(alpha, nelements(dg, cache))
    if alpha_smooth
        resize!(alpha_tmp, nelements(dg, cache))
    end

    # magic parameters
    # TODO: Are there better values for Float32?
    RealT = real(dg)
    threshold = 0.5f0 * 10^(convert(RealT, -1.8) * nnodes(dg)^convert(RealT, 0.25))
    o_0001 = convert(RealT, 0.0001)
    parameter_s = log((1 - o_0001) / o_0001)

    @threaded for element in eachelement(dg, cache)
        # This is dispatched by mesh dimension.
        # Use this function barrier and unpack inside to avoid passing closures to
        # Polyester.jl with `@batch` (`@threaded`).
        # Otherwise, `@threaded` does not work here with Julia ARM on macOS.
        # See https://github.com/JuliaSIMD/Polyester.jl/issues/88.
        calc_indicator_hennemann_gassner!(indicator_hg, threshold, parameter_s, u,
                                          element, mesh, equations, dg, cache)
    end

    if alpha_smooth
        apply_smoothing!(mesh, alpha, alpha_tmp, dg, cache)
    end

    return alpha
end

"""
    IndicatorLöhner (equivalent to IndicatorLoehner)

    IndicatorLöhner(equations::AbstractEquations, basis;
                    f_wave=0.2, variable)
    IndicatorLöhner(semi::AbstractSemidiscretization;
                    f_wave=0.2, variable)

AMR indicator adapted from a FEM indicator by Löhner (1987), also used in the
FLASH code as standard AMR indicator.
The indicator estimates a weighted second derivative of a specified variable locally.

When constructed to be used for AMR, pass the `semi`. Pass the `equations`,
and `basis` if this indicator should be used for shock capturing.

## References

- Löhner (1987)
  "An adaptive finite element scheme for transient problems in CFD"
  [doi: 10.1016/0045-7825(87)90098-3](https://doi.org/10.1016/0045-7825(87)90098-3)
- [https://flash.rochester.edu/site/flashcode/user_support/flash4_ug_4p62/node59.html#SECTION05163100000000000000](https://flash.rochester.edu/site/flashcode/user_support/flash4_ug_4p62/node59.html#SECTION05163100000000000000)
"""
struct IndicatorLöhner{RealT <: Real, Variable, Cache} <: AbstractIndicator
    f_wave::RealT # TODO: Taal documentation
    variable::Variable
    cache::Cache
end

# this method is used when the indicator is constructed as for shock-capturing volume integrals
function IndicatorLöhner(equations::AbstractEquations, basis;
                         f_wave = 0.2, variable)
    cache = create_cache(IndicatorLöhner, equations, basis)
    IndicatorLöhner{typeof(f_wave), typeof(variable), typeof(cache)}(f_wave, variable,
                                                                     cache)
end

# this method is used when the indicator is constructed as for AMR
function IndicatorLöhner(semi::AbstractSemidiscretization;
                         f_wave = 0.2, variable)
    cache = create_cache(IndicatorLöhner, semi)
    IndicatorLöhner{typeof(f_wave), typeof(variable), typeof(cache)}(f_wave, variable,
                                                                     cache)
end

function Base.show(io::IO, indicator::IndicatorLöhner)
    @nospecialize indicator # reduce precompilation time

    print(io, "IndicatorLöhner(")
    print(io, "f_wave=", indicator.f_wave, ", variable=", indicator.variable, ")")
end

function Base.show(io::IO, ::MIME"text/plain", indicator::IndicatorLöhner)
    @nospecialize indicator # reduce precompilation time

    if get(io, :compact, false)
        show(io, indicator)
    else
        setup = [
            "indicator variable" => indicator.variable,
            "f_wave" => indicator.f_wave
        ]
        summary_box(io, "IndicatorLöhner", setup)
    end
end

const IndicatorLoehner = IndicatorLöhner

# dirty Löhner estimate, direction by direction, assuming constant nodes
@inline function local_löhner_estimate(um::Real, u0::Real, up::Real,
                                       löhner::IndicatorLöhner)
    num = abs(up - 2 * u0 + um)
    den = abs(up - u0) + abs(u0 - um) +
          löhner.f_wave * (abs(up) + 2 * abs(u0) + abs(um))
    return num / den
end

"""
    IndicatorMax(equations::AbstractEquations, basis; variable)
    IndicatorMax(semi::AbstractSemidiscretization; variable)

A simple indicator returning the maximum of `variable` in an element.
When constructed to be used for AMR, pass the `semi`. Pass the `equations`,
and `basis` if this indicator should be used for shock capturing.
"""
struct IndicatorMax{Variable, Cache <: NamedTuple} <: AbstractIndicator
    variable::Variable
    cache::Cache
end

# this method is used when the indicator is constructed as for shock-capturing volume integrals
function IndicatorMax(equations::AbstractEquations, basis;
                      variable)
    cache = create_cache(IndicatorMax, equations, basis)
    IndicatorMax{typeof(variable), typeof(cache)}(variable, cache)
end

# this method is used when the indicator is constructed as for AMR
function IndicatorMax(semi::AbstractSemidiscretization;
                      variable)
    cache = create_cache(IndicatorMax, semi)
    return IndicatorMax{typeof(variable), typeof(cache)}(variable, cache)
end

function Base.show(io::IO, indicator::IndicatorMax)
    @nospecialize indicator # reduce precompilation time

    print(io, "IndicatorMax(")
    print(io, "variable=", indicator.variable, ")")
end

function Base.show(io::IO, ::MIME"text/plain", indicator::IndicatorMax)
    @nospecialize indicator # reduce precompilation time

    if get(io, :compact, false)
        show(io, indicator)
    else
        setup = [
            "indicator variable" => indicator.variable
        ]
        summary_box(io, "IndicatorMax", setup)
    end
end
end # @muladd
