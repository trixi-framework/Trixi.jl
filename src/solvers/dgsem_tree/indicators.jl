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

    if get(io, :compact, false)
        show(io, indicator)
    else
        setup = [
            "indicator variable" => indicator.variable,
            "max. α" => indicator.alpha_max,
            "min. α" => indicator.alpha_min,
            "smooth α" => (indicator.alpha_smooth ? "yes" : "no"),
        ]
        summary_box(io, "IndicatorHennemannGassner", setup)
    end
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
    threshold = 0.5 * 10^(-1.8 * (nnodes(dg))^0.25)
    parameter_s = log((1 - 0.0001) / 0.0001)

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
- http://flash.uchicago.edu/site/flashcode/user_support/flash4_ug_4p62/node59.html#SECTION05163100000000000000
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
            "f_wave" => indicator.f_wave,
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

abstract type AbstractSubcellLimiter end

function create_cache(typ::Type{LimiterType},
                      semi) where {LimiterType <: AbstractSubcellLimiter}
    create_cache(typ, mesh_equations_solver_cache(semi)...)
end

function get_element_variables!(element_variables, limiter::AbstractSubcellLimiter,
                                ::VolumeIntegralSubcellLimiting)
    element_variables[:smooth_indicator_elementwise] = limiter.IndicatorHG.cache.alpha
    return nothing
end

"""
    SubcellLimiterIDP(equations::AbstractEquations, basis;
                      local_minmax_variables_cons = [],
                      positivity_variables_cons = [],
                      positivity_variables_nonlinear = (),
                      positivity_correction_factor = 0.1,
                      spec_entropy = false,
                      math_entropy = false,
                      bar_states = true,
                      max_iterations_newton = 10,
                      newton_tolerances = (1.0e-12, 1.0e-14),
                      gamma_constant_newton = 2 * ndims(equations),
                      smoothness_indicator = false,
                      threshold_smoothness_indicator = 0.1,
                      variable_smoothness_indicator = density_pressure)

Subcell invariant domain preserving (IDP) limiting used with [`VolumeIntegralSubcellLimiting`](@ref)
including:
- maximum/minimum Zalesak-type limiting for conservative variables (`local_minmax_variables_cons`)
- positivity limiting for conservative (`positivity_variables_cons`) and non-linear variables (`positivity_variables_nonlinear`)
- one-sided limiting for specific and mathematical entropy (`spec_entropy`, `math_entropy`)

The bounds can be calculated using the `bar_states` or the low-order FV solution. The positivity
limiter uses `positivity_correction_factor` such that `u^new >= positivity_correction_factor * u^FV`.
The Newton-bisection method for the limiting of non-linear variables uses maximal `max_iterations_newton`
iterations, tolerances `newton_tolerances` and the gamma constant `gamma_constant_newton`
(gamma_constant_newton>=2*d, where d=#dimensions).

A hard-switch [`IndicatorHennemannGassner`](@ref) can be activated (`smoothness_indicator`) with
`variable_smoothness_indicator`, which disables subcell blending for element-wise
indicator values <= `threshold_smoothness_indicator`.

## References

- Rueda-Ramírez, Pazner, Gassner (2022)
  Subcell Limiting Strategies for Discontinuous Galerkin Spectral Element Methods
  [DOI: 10.1016/j.compfluid.2022.105627](https://doi.org/10.1016/j.compfluid.2022.105627)
- Pazner (2020)
  Sparse invariant domain preserving discontinuous Galerkin methods with subcell convex limiting
  [DOI: 10.1016/j.cma.2021.113876](https://doi.org/10.1016/j.cma.2021.113876)

!!! warning "Experimental implementation"
    This is an experimental feature and may change in future releases.
"""
struct SubcellLimiterIDP{RealT <: Real, LimitingVariablesNonlinear,
                         Cache, Indicator} <: AbstractSubcellLimiter
    local_minmax::Bool
    local_minmax_variables_cons::Vector{Int}                   # Local mininum/maximum principles for conservative variables
    positivity::Bool
    positivity_variables_cons::Vector{Int}                     # Positivity for conservative variables
    positivity_variables_nonlinear::LimitingVariablesNonlinear # Positivity for nonlinear variables
    positivity_correction_factor::RealT
    spec_entropy::Bool
    math_entropy::Bool
    bar_states::Bool
    cache::Cache
    max_iterations_newton::Int
    newton_tolerances::Tuple{RealT, RealT}          # Relative and absolute tolerances for Newton's method
    gamma_constant_newton::RealT                    # Constant for the subcell limiting of convex (nonlinear) constraints
    smoothness_indicator::Bool
    threshold_smoothness_indicator::RealT
    IndicatorHG::Indicator
end

# this method is used when the limiter is constructed as for shock-capturing volume integrals
function SubcellLimiterIDP(equations::AbstractEquations, basis;
                           local_minmax_variables_cons = [],
                           positivity_variables_cons = [],
                           positivity_variables_nonlinear = (),
                           positivity_correction_factor = 0.1,
                           spec_entropy = false,
                           math_entropy = false,
                           bar_states = true,
                           max_iterations_newton = 10,
                           newton_tolerances = (1.0e-12, 1.0e-14),
                           gamma_constant_newton = 2 * ndims(equations),
                           smoothness_indicator = false,
                           threshold_smoothness_indicator = 0.1,
                           variable_smoothness_indicator = density_pressure)
    local_minmax = (length(local_minmax_variables_cons) > 0)
    positivity = (length(positivity_variables_cons) +
                  length(positivity_variables_nonlinear) > 0)
    if math_entropy && spec_entropy
        error("Only one of the two can be selected: math_entropy/spec_entropy")
    end

    number_bounds = 2 * length(local_minmax_variables_cons) +
                    length(positivity_variables_nonlinear) +
                    spec_entropy + math_entropy

    for index in positivity_variables_cons
        if !(index in local_minmax_variables_cons)
            number_bounds += 1
        end
    end

    cache = create_cache(SubcellLimiterIDP, equations, basis, number_bounds, bar_states)

    if smoothness_indicator
        IndicatorHG = IndicatorHennemannGassner(equations, basis, alpha_max = 1.0,
                                                alpha_smooth = false,
                                                variable = variable_smoothness_indicator)
    else
        IndicatorHG = nothing
    end
    SubcellLimiterIDP{typeof(positivity_correction_factor),
                      typeof(positivity_variables_nonlinear),
                      typeof(cache), typeof(IndicatorHG)}(local_minmax,
                                                          local_minmax_variables_cons,
                                                          positivity,
                                                          positivity_variables_cons,
                                                          positivity_variables_nonlinear,
                                                          positivity_correction_factor,
                                                          spec_entropy,
                                                          math_entropy,
                                                          bar_states,
                                                          cache,
                                                          max_iterations_newton,
                                                          newton_tolerances,
                                                          gamma_constant_newton,
                                                          smoothness_indicator,
                                                          threshold_smoothness_indicator,
                                                          IndicatorHG)
end

function Base.show(io::IO, limiter::SubcellLimiterIDP)
    @nospecialize limiter # reduce precompilation time
    @unpack local_minmax, positivity, spec_entropy, math_entropy = limiter

    print(io, "SubcellLimiterIDP(")
    if !(local_minmax || positivity || spec_entropy || math_entropy)
        print(io, "No limiter selected => pure DG method")
    else
        print(io, "limiter=(")
        local_minmax && print(io, "min/max limiting, ")
        positivity && print(io, "positivity, ")
        spec_entropy && print(io, "specific entropy, ")
        math_entropy && print(io, "mathematical entropy, ")
        print(io, "), ")
    end
    limiter.smoothness_indicator &&
        print(io, ", Smoothness indicator: ", limiter.IndicatorHG,
              " with threshold ", limiter.threshold_smoothness_indicator, "), ")
    print(io,
          "Local bounds with $(limiter.bar_states ? "Bar States" : "FV solution")")
    print(io, ")")
end

function Base.show(io::IO, ::MIME"text/plain", limiter::SubcellLimiterIDP)
    @nospecialize limiter # reduce precompilation time
    @unpack local_minmax, positivity, spec_entropy, math_entropy = limiter

    if get(io, :compact, false)
        show(io, limiter)
    else
        if !(local_minmax || positivity || spec_entropy || math_entropy)
            setup = ["limiter" => "No limiter selected => pure DG method"]
        else
            setup = ["limiter" => ""]
            if local_minmax
                setup = [
                    setup...,
                    "" => "local maximum/minimum bounds for conservative variables $(limiter.local_minmax_variables_cons)",
                ]
            end
            if positivity
                string = "positivity for conservative variables $(limiter.positivity_variables_cons) and $(limiter.positivity_variables_nonlinear)"
                setup = [setup..., "" => string]
                setup = [
                    setup...,
                    "" => "   positivity correction factor = $(limiter.positivity_correction_factor)",
                ]
            end
            if spec_entropy
                setup = [setup..., "" => "local minimum bound for specific entropy"]
            end
            if math_entropy
                setup = [setup..., "" => "local maximum bound for mathematical entropy"]
            end
            setup = [
                setup...,
                "Local bounds" => (limiter.bar_states ? "Bar States" : "FV solution"),
            ]
            if limiter.smoothness_indicator
                setup = [
                    setup...,
                    "Smoothness indicator" => "$(limiter.IndicatorHG) using threshold $(limiter.threshold_smoothness_indicator)",
                ]
            end
            summary_box(io, "SubcellLimiterIDP", setup)
        end
    end
end

function get_node_variables!(node_variables, limiter::SubcellLimiterIDP,
                             ::VolumeIntegralSubcellLimiting, equations)
    node_variables[:alpha_limiter] = limiter.cache.container_subcell_limiter.alpha
    # TODO: alpha is not filled before the first timestep.
    return nothing
end

"""
    SubcellLimiterMCL(equations::AbstractEquations, basis;
                      DensityLimiter = true,
                      DensityAlphaForAll = false,
                      SequentialLimiter = true,
                      ConservativeLimiter = false,
                      PressurePositivityLimiterKuzmin = false,
                      PressurePositivityLimiterKuzminExact = true,
                      DensityPositivityLimiter = false,
                      DensityPositivityCorrectionFactor = 0.0,
                      SemiDiscEntropyLimiter = false,
                      smoothness_indicator = false,
                      threshold_smoothness_indicator = 0.1,
                      variable_smoothness_indicator = density_pressure,
                      Plotting = true)

Subcell monolithic convex limiting (MCL) used with [`VolumeIntegralSubcellLimiting`](@ref) including:
- local two-sided limiting for `cons(1)` (`DensityLimiter`)
- transfer amount of `DensityLimiter` to all quantities (`DensityAlphaForAll`)
- local two-sided limiting for variables `phi:=cons(i)/cons(1)` (`SequentialLimiter`)
- local two-sided limiting for conservative variables (`ConservativeLimiter`)
- positivity limiting for `cons(1)` (`DensityPositivityLimiter`) and pressure (`PressurePositivityLimiterKuzmin`)
- semidiscrete entropy fix (`SemiDiscEntropyLimiter`)

The pressure positivity limiting preserves a sharp version (`PressurePositivityLimiterKuzminExact`)
and a more cautious one. The density positivity limiter uses a `DensityPositivityCorrectionFactor`
such that `u^new >= positivity_correction_factor * u^FV`. All additional analyses for plotting routines
can be disabled via `Plotting=false` (see `save_alpha` and `update_alpha_max_avg!`).

A hard-switch [`IndicatorHennemannGassner`](@ref) can be activated (`smoothness_indicator`) with
`variable_smoothness_indicator`, which disables subcell blending for element-wise
indicator values <= `threshold_smoothness_indicator`.

## References

- Rueda-Ramírez, Bolm, Kuzmin, Gassner (2023)
  Monolithic Convex Limiting for Legendre-Gauss-Lobatto Discontinuous Galerkin Spectral Element Methods
  [arXiv:2303.00374](https://doi.org/10.48550/arXiv.2303.00374)
- Kuzmin (2020)
  Monolithic convex limiting for continuous finite element discretizations of hyperbolic conservation laws
  [DOI: 10.1016/j.cma.2019.112804](https://doi.org/10.1016/j.cma.2019.112804)

!!! warning "Experimental implementation"
    This is an experimental feature and may change in future releases.
"""
struct SubcellLimiterMCL{RealT <: Real, Cache, Indicator} <: AbstractSubcellLimiter
    cache::Cache
    DensityLimiter::Bool        # Impose local maximum/minimum for cons(1) based on bar states
    DensityAlphaForAll::Bool    # Use the cons(1) blending coefficient for all quantities
    SequentialLimiter::Bool     # Impose local maximum/minimum for variables phi:=cons(i)/cons(1) i 2:nvariables based on bar states
    ConservativeLimiter::Bool   # Impose local maximum/minimum for conservative variables 2:nvariables based on bar states
    PressurePositivityLimiterKuzmin::Bool       # Impose positivity for pressure â la Kuzmin
    PressurePositivityLimiterKuzminExact::Bool  # Only for PressurePositivityLimiterKuzmin=true: Use the exact calculation of alpha
    DensityPositivityLimiter::Bool              # Impose positivity for cons(1)
    DensityPositivityCorrectionFactor::RealT    # Correction Factor for DensityPositivityLimiter in [0,1)
    SemiDiscEntropyLimiter::Bool                # synchronized semidiscrete entropy fix
    smoothness_indicator::Bool                  # activates smoothness indicator: IndicatorHennemannGassner
    threshold_smoothness_indicator::RealT       # threshold for smoothness indicator
    IndicatorHG::Indicator
    Plotting::Bool
end

# this method is used when the limiter is constructed as for shock-capturing volume integrals
function SubcellLimiterMCL(equations::AbstractEquations, basis;
                           DensityLimiter = true,
                           DensityAlphaForAll = false,
                           SequentialLimiter = true,
                           ConservativeLimiter = false,
                           PressurePositivityLimiterKuzmin = false,
                           PressurePositivityLimiterKuzminExact = true,
                           DensityPositivityLimiter = false,
                           DensityPositivityCorrectionFactor = 0.0,
                           SemiDiscEntropyLimiter = false,
                           smoothness_indicator = false,
                           threshold_smoothness_indicator = 0.1,
                           variable_smoothness_indicator = density_pressure,
                           Plotting = true)
    if SequentialLimiter && ConservativeLimiter
        error("Only one of the two can be selected: SequentialLimiter/ConservativeLimiter")
    end
    cache = create_cache(SubcellLimiterMCL, equations, basis,
                         PressurePositivityLimiterKuzmin)
    if smoothness_indicator
        IndicatorHG = IndicatorHennemannGassner(equations, basis, alpha_smooth = false,
                                                variable = variable_smoothness_indicator)
    else
        IndicatorHG = nothing
    end
    SubcellLimiterMCL{typeof(threshold_smoothness_indicator), typeof(cache),
                      typeof(IndicatorHG)}(cache,
                                           DensityLimiter, DensityAlphaForAll,
                                           SequentialLimiter, ConservativeLimiter,
                                           PressurePositivityLimiterKuzmin,
                                           PressurePositivityLimiterKuzminExact,
                                           DensityPositivityLimiter,
                                           DensityPositivityCorrectionFactor,
                                           SemiDiscEntropyLimiter,
                                           smoothness_indicator,
                                           threshold_smoothness_indicator, IndicatorHG,
                                           Plotting)
end

function Base.show(io::IO, limiter::SubcellLimiterMCL)
    @nospecialize limiter # reduce precompilation time

    print(io, "SubcellLimiterMCL(")
    limiter.DensityLimiter && print(io, "; dens")
    limiter.DensityAlphaForAll && print(io, "; dens alpha ∀")
    limiter.SequentialLimiter && print(io, "; seq")
    limiter.ConservativeLimiter && print(io, "; cons")
    if limiter.PressurePositivityLimiterKuzmin
        print(io,
              "; $(limiter.PressurePositivityLimiterKuzminExact ? "pres (Kuzmin ex)" : "pres (Kuzmin)")")
    end
    limiter.DensityPositivityLimiter && print(io, "; dens pos")
    if limiter.DensityPositivityCorrectionFactor != 0
        print(io,
              " with correction factor $(limiter.DensityPositivityCorrectionFactor)")
    end
    limiter.SemiDiscEntropyLimiter && print(io, "; semid. entropy")
    limiter.smoothness_indicator &&
        print(io, "; Smoothness indicator: ", limiter.IndicatorHG,
              " with threshold ", limiter.threshold_smoothness_indicator)
    print(io, ")")
end

function Base.show(io::IO, ::MIME"text/plain", limiter::SubcellLimiterMCL)
    @nospecialize limiter # reduce precompilation time
    @unpack DensityLimiter, DensityAlphaForAll, SequentialLimiter, ConservativeLimiter,
    PressurePositivityLimiterKuzminExact, DensityPositivityLimiter, SemiDiscEntropyLimiter = limiter

    if get(io, :compact, false)
        show(io, limiter)
    else
        setup = ["limiter" => ""]
        DensityLimiter && (setup = [setup..., "" => "DensityLimiter"])
        DensityAlphaForAll && (setup = [setup..., "" => "DensityAlphaForAll"])
        SequentialLimiter && (setup = [setup..., "" => "SequentialLimiter"])
        ConservativeLimiter && (setup = [setup..., "" => "ConservativeLimiter"])
        if limiter.PressurePositivityLimiterKuzmin
            setup = [
                setup...,
                "" => "PressurePositivityLimiterKuzmin $(PressurePositivityLimiterKuzminExact ? "(exact)" : "")",
            ]
        end
        if DensityPositivityLimiter
            if limiter.DensityPositivityCorrectionFactor != 0.0
                setup = [
                    setup...,
                    "" => "DensityPositivityLimiter with correction factor $(limiter.DensityPositivityCorrectionFactor)",
                ]
            else
                setup = [setup..., "" => "DensityPositivityLimiter"]
            end
        end
        SemiDiscEntropyLimiter && (setup = [setup..., "" => "SemiDiscEntropyLimiter"])
        if limiter.smoothness_indicator
            setup = [
                setup...,
                "Smoothness indicator" => "$(limiter.IndicatorHG) using threshold $(limiter.threshold_smoothness_indicator)",
            ]
        end
        summary_box(io, "SubcellLimiterMCL", setup)
    end
end

function get_node_variables!(node_variables, limiter::SubcellLimiterMCL,
                             ::VolumeIntegralSubcellLimiting, equations)
    if !limiter.Plotting
        return nothing
    end
    @unpack alpha = limiter.cache.container_subcell_limiter
    variables = varnames(cons2cons, equations)
    for v in eachvariable(equations)
        s = Symbol("alpha_", variables[v])
        node_variables[s] = alpha[v, ntuple(_ -> :, size(alpha, 2) + 1)...]
    end

    if limiter.PressurePositivityLimiterKuzmin
        @unpack alpha_pressure = limiter.cache.container_subcell_limiter
        node_variables[:alpha_pressure] = alpha_pressure
    end

    if limiter.SemiDiscEntropyLimiter
        @unpack alpha_entropy = limiter.cache.container_subcell_limiter
        node_variables[:alpha_entropy] = alpha_entropy
    end

    for v in eachvariable(equations)
        @unpack alpha_mean = limiter.cache.container_subcell_limiter
        s = Symbol("alpha_mean_", variables[v])
        node_variables[s] = copy(alpha_mean[v, ntuple(_ -> :, size(alpha, 2) + 1)...])
    end

    if limiter.PressurePositivityLimiterKuzmin
        @unpack alpha_mean_pressure = limiter.cache.container_subcell_limiter
        node_variables[:alpha_mean_pressure] = alpha_mean_pressure
    end

    if limiter.SemiDiscEntropyLimiter
        @unpack alpha_mean_entropy = limiter.cache.container_subcell_limiter
        node_variables[:alpha_mean_entropy] = alpha_mean_entropy
    end

    return nothing
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
            "indicator variable" => indicator.variable,
        ]
        summary_box(io, "IndicatorMax", setup)
    end
end

"""
    IndicatorNeuralNetwork

Artificial neural network based indicator used for shock-capturing or AMR.
Depending on the indicator_type, different input values and corresponding trained networks are used.

`indicator_type = NeuralNetworkPerssonPeraire()`
- Input: The energies in lower modes as well as nnodes(dg).

`indicator_type = NeuralNetworkRayHesthaven()`
- 1d Input: Cell average of the cell and its neighboring cells as well as the interface values.
- 2d Input: Linear modal values of the cell and its neighboring cells.

- Ray, Hesthaven (2018)
  "An artificial neural network as a troubled-cell indicator"
  [doi:10.1016/j.jcp.2018.04.029](https://doi.org/10.1016/j.jcp.2018.04.029)
- Ray, Hesthaven (2019)
  "Detecting troubled-cells on two-dimensional unstructured grids using a neural network"
  [doi:10.1016/j.jcp.2019.07.043](https://doi.org/10.1016/j.jcp.2019.07.043)

`indicator_type = CNN (Only in 2d)`
- Based on convolutional neural network.
- 2d Input: Interpolation of the nodal values of the `indicator.variable` to the 4x4 LGL nodes.

If `alpha_continuous == true` the continuous network output for troubled cells (`alpha > 0.5`) is considered.
If the cells are good (`alpha < 0.5`), `alpha` is set to `0`.
If `alpha_continuous == false`, the blending factor is set to `alpha = 0` for good cells and
`alpha = 1` for troubled cells.

!!! warning "Experimental implementation"
    This is an experimental feature and may change in future releases.

"""
struct IndicatorNeuralNetwork{IndicatorType, RealT <: Real, Variable, Chain, Cache} <:
       AbstractIndicator
    indicator_type::IndicatorType
    alpha_max::RealT
    alpha_min::RealT
    alpha_smooth::Bool
    alpha_continuous::Bool
    alpha_amr::Bool
    variable::Variable
    network::Chain
    cache::Cache
end

# this method is used when the indicator is constructed as for shock-capturing volume integrals
function IndicatorNeuralNetwork(equations::AbstractEquations, basis;
                                indicator_type,
                                alpha_max = 0.5,
                                alpha_min = 0.001,
                                alpha_smooth = true,
                                alpha_continuous = true,
                                alpha_amr = false,
                                variable,
                                network)
    alpha_max, alpha_min = promote(alpha_max, alpha_min)
    IndicatorType = typeof(indicator_type)
    cache = create_cache(IndicatorNeuralNetwork{IndicatorType}, equations, basis)
    IndicatorNeuralNetwork{IndicatorType, typeof(alpha_max), typeof(variable),
                           typeof(network), typeof(cache)}(indicator_type, alpha_max,
                                                           alpha_min, alpha_smooth,
                                                           alpha_continuous, alpha_amr,
                                                           variable,
                                                           network, cache)
end

# this method is used when the indicator is constructed as for AMR
function IndicatorNeuralNetwork(semi::AbstractSemidiscretization;
                                indicator_type,
                                alpha_max = 0.5,
                                alpha_min = 0.001,
                                alpha_smooth = true,
                                alpha_continuous = true,
                                alpha_amr = true,
                                variable,
                                network)
    alpha_max, alpha_min = promote(alpha_max, alpha_min)
    IndicatorType = typeof(indicator_type)
    cache = create_cache(IndicatorNeuralNetwork{IndicatorType}, semi)
    IndicatorNeuralNetwork{IndicatorType, typeof(alpha_max), typeof(variable),
                           typeof(network), typeof(cache)}(indicator_type, alpha_max,
                                                           alpha_min, alpha_smooth,
                                                           alpha_continuous, alpha_amr,
                                                           variable,
                                                           network, cache)
end

function Base.show(io::IO, indicator::IndicatorNeuralNetwork)
    @nospecialize indicator # reduce precompilation time

    print(io, "IndicatorNeuralNetwork(")
    print(io, indicator.indicator_type)
    print(io, ", alpha_max=", indicator.alpha_max)
    print(io, ", alpha_min=", indicator.alpha_min)
    print(io, ", alpha_smooth=", indicator.alpha_smooth)
    print(io, ", alpha_continuous=", indicator.alpha_continuous)
    print(io, indicator.variable)
    print(io, ")")
end

function Base.show(io::IO, ::MIME"text/plain", indicator::IndicatorNeuralNetwork)
    @nospecialize indicator # reduce precompilation time

    if get(io, :compact, false)
        show(io, indicator)
    else
        setup = [
            "indicator type" => indicator.indicator_type,
            "max. α" => indicator.alpha_max,
            "min. α" => indicator.alpha_min,
            "smooth α" => (indicator.alpha_smooth ? "yes" : "no"),
            "continuous α" => (indicator.alpha_continuous ? "yes" : "no"),
            "indicator variable" => indicator.variable,
        ]
        summary_box(io, "IndicatorNeuralNetwork", setup)
    end
end

# Convert probability for troubled cell to indicator value for shockcapturing/AMR
@inline function probability_to_indicator(probability_troubled_cell, alpha_continuous,
                                          alpha_amr,
                                          alpha_min, alpha_max)
    # Initialize indicator to zero
    alpha_element = zero(probability_troubled_cell)

    if alpha_continuous && !alpha_amr
        # Set good cells to 0 and troubled cells to continuous value of the network prediction
        if probability_troubled_cell > 0.5
            alpha_element = probability_troubled_cell
        else
            alpha_element = zero(probability_troubled_cell)
        end

        # Take care of the case close to pure FV
        if alpha_element > 1 - alpha_min
            alpha_element = one(alpha_element)
        end

        # Scale the probability for a troubled cell (in [0,1]) to the maximum allowed alpha
        alpha_element *= alpha_max
    elseif !alpha_continuous && !alpha_amr
        # Set good cells to 0 and troubled cells to 1
        if probability_troubled_cell > 0.5
            alpha_element = alpha_max
        else
            alpha_element = zero(alpha_max)
        end
    elseif alpha_amr
        # The entire continuous output of the neural network is used for AMR
        alpha_element = probability_troubled_cell

        # Scale the probability for a troubled cell (in [0,1]) to the maximum allowed alpha
        alpha_element *= alpha_max
    end

    return alpha_element
end

"""
    NeuralNetworkPerssonPeraire

Indicator type for creating an `IndicatorNeuralNetwork` indicator.

!!! warning "Experimental implementation"
    This is an experimental feature and may change in future releases.

See also: [`IndicatorNeuralNetwork`](@ref)
"""
struct NeuralNetworkPerssonPeraire end

"""
    NeuralNetworkRayHesthaven

Indicator type for creating an `IndicatorNeuralNetwork` indicator.

!!! warning "Experimental implementation"
    This is an experimental feature and may change in future releases.

See also: [`IndicatorNeuralNetwork`](@ref)
"""
struct NeuralNetworkRayHesthaven end

"""
    NeuralNetworkCNN

Indicator type for creating an `IndicatorNeuralNetwork` indicator.

!!! warning "Experimental implementation"
    This is an experimental feature and may change in future releases.

See also: [`IndicatorNeuralNetwork`](@ref)
"""
struct NeuralNetworkCNN end
end # @muladd
