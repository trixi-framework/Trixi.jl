# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

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

!!! note
    This limiter and the correction callback [`SubcellLimiterIDPCorrection`](@ref) only work together.
    Without the callback, no limiting takes place, leading to a standard flux-differencing DGSEM scheme.

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

    bound_keys = ()
    if local_minmax
        for i in local_minmax_variables_cons
            bound_keys = (bound_keys..., Symbol("$(i)_min"), Symbol("$(i)_max"))
        end
    end
    if spec_entropy
        bound_keys = (bound_keys..., :spec_entropy_min)
    end
    if math_entropy
        bound_keys = (bound_keys..., :math_entropy_max)
    end
    for i in positivity_variables_cons
        if !(i in local_minmax_variables_cons)
            bound_keys = (bound_keys..., Symbol("$(i)_min"))
        end
    end
    for variable in positivity_variables_nonlinear
        bound_keys = (bound_keys..., Symbol("$(variable)_min"))
    end

    cache = create_cache(SubcellLimiterIDP, equations, basis, bound_keys, bar_states)

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
    node_variables[:alpha_limiter] = limiter.cache.subcell_limiter_coefficients.alpha
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
    @unpack alpha = limiter.cache.subcell_limiter_coefficients
    variables = varnames(cons2cons, equations)
    for v in eachvariable(equations)
        s = Symbol("alpha_", variables[v])
        node_variables[s] = alpha[v, ntuple(_ -> :, size(alpha, 2) + 1)...]
    end

    if limiter.PressurePositivityLimiterKuzmin
        @unpack alpha_pressure = limiter.cache.subcell_limiter_coefficients
        node_variables[:alpha_pressure] = alpha_pressure
    end

    if limiter.SemiDiscEntropyLimiter
        @unpack alpha_entropy = limiter.cache.subcell_limiter_coefficients
        node_variables[:alpha_entropy] = alpha_entropy
    end

    for v in eachvariable(equations)
        @unpack alpha_mean = limiter.cache.subcell_limiter_coefficients
        s = Symbol("alpha_mean_", variables[v])
        node_variables[s] = copy(alpha_mean[v, ntuple(_ -> :, size(alpha, 2) + 1)...])
    end

    if limiter.PressurePositivityLimiterKuzmin
        @unpack alpha_mean_pressure = limiter.cache.subcell_limiter_coefficients
        node_variables[:alpha_mean_pressure] = alpha_mean_pressure
    end

    if limiter.SemiDiscEntropyLimiter
        @unpack alpha_mean_entropy = limiter.cache.subcell_limiter_coefficients
        node_variables[:alpha_mean_entropy] = alpha_mean_entropy
    end

    return nothing
end
end # @muladd
