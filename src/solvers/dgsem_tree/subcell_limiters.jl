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
                      local_twosided_variables_cons = String[],
                      positivity_variables_cons = String[],
                      positivity_variables_nonlinear = [],
                      positivity_correction_factor = 0.1,
                      local_onesided_variables_nonlinear = [],
                      bar_states = true,
                      max_iterations_newton = 10,
                      newton_tolerances = (1.0e-12, 1.0e-14),
                      gamma_constant_newton = 2 * ndims(equations),
                      smoothness_indicator = false,
                      threshold_smoothness_indicator = 0.1,
                      variable_smoothness_indicator = density_pressure)

Subcell invariant domain preserving (IDP) limiting used with [`VolumeIntegralSubcellLimiting`](@ref)
including:
- Local two-sided Zalesak-type limiting for conservative variables (`local_twosided_variables_cons`)
- Positivity limiting for conservative variables (`positivity_variables_cons`) and nonlinear variables
(`positivity_variables_nonlinear`)
- Local one-sided limiting for nonlinear variables, e.g. `entropy_guermond_etal` and `entropy_math`
with `local_onesided_variables_nonlinear`

To use these three limiting options use the following structure:

***Conservative variables*** to be limited are passed as a vector of strings, e.g.
`local_twosided_variables_cons = ["rho"]` and `positivity_variables_cons = ["rho"]`.
For ***nonlinear variables***, the wanted variable functions are passed within a vector: To ensure
positivity use a plain vector including the desired variables, e.g. `positivity_variables_nonlinear = [pressure]`.
For local one-sided limiting pass the variable function combined with the requested bound
(`min` or `max`) as a tuple. For instance, to impose a lower local bound on the modified specific
entropy by Guermond et al. use `local_onesided_variables_nonlinear = [(Trixi.entropy_guermond_etal, min)]`.

The bounds can be calculated using the `bar_states` or the low-order FV solution. The positivity
limiter uses `positivity_correction_factor` such that `u^new >= positivity_correction_factor * u^FV`.
Local and global limiting of nonlinear variables uses a Newton-bisection method with a maximum of
`max_iterations_newton` iterations, relative and absolute tolerances of `newton_tolerances`
and a provisional update constant `gamma_constant_newton` (`gamma_constant_newton>=2*d`,
where `d = #dimensions`). See equation (20) of Pazner (2020) and equation (30) of Rueda-Ramírez et al. (2022).

A hard-switch [`IndicatorHennemannGassner`](@ref) can be activated (`smoothness_indicator`) with
`variable_smoothness_indicator`, which disables subcell blending for element-wise
indicator values <= `threshold_smoothness_indicator`.

!!! note
    This limiter and the correction callback [`SubcellLimiterIDPCorrection`](@ref) only work together.
    Without the callback, no correction takes place, leading to a standard low-order FV scheme.

## References

- Rueda-Ramírez, Pazner, Gassner (2022)
  Subcell Limiting Strategies for Discontinuous Galerkin Spectral Element Methods
  [DOI: 10.1016/j.compfluid.2022.105627](https://doi.org/10.1016/j.compfluid.2022.105627)
- Pazner (2020)
  Sparse invariant domain preserving discontinuous Galerkin methods with subcell convex limiting
  [DOI: 10.1016/j.cma.2021.113876](https://doi.org/10.1016/j.cma.2021.113876)
"""
struct SubcellLimiterIDP{RealT <: Real, LimitingVariablesNonlinear,
                         LimitingOnesidedVariablesNonlinear, Cache,
                         Indicator} <: AbstractSubcellLimiter
    local_twosided::Bool
    local_twosided_variables_cons::Vector{Int}                 # Local two-sided limiting for conservative variables
    positivity::Bool
    positivity_variables_cons::Vector{Int}                     # Positivity for conservative variables
    positivity_variables_nonlinear::LimitingVariablesNonlinear # Positivity for nonlinear variables
    positivity_correction_factor::RealT
    local_onesided::Bool
    local_onesided_variables_nonlinear::LimitingOnesidedVariablesNonlinear # Local one-sided limiting for nonlinear variables
    bar_states::Bool
    cache::Cache
    max_iterations_newton::Int
    newton_tolerances::Tuple{RealT, RealT}  # Relative and absolute tolerances for Newton's method
    gamma_constant_newton::RealT            # Constant for the subcell limiting of convex (nonlinear) constraints
    smoothness_indicator::Bool
    threshold_smoothness_indicator::RealT
    IndicatorHG::Indicator
end

# this method is used when the limiter is constructed as for shock-capturing volume integrals
function SubcellLimiterIDP(equations::AbstractEquations, basis;
                           local_twosided_variables_cons = String[],
                           positivity_variables_cons = String[],
                           positivity_variables_nonlinear = [],
                           positivity_correction_factor = 0.1,
                           local_onesided_variables_nonlinear = [],
                           bar_states = true,
                           max_iterations_newton = 10,
                           newton_tolerances = (1.0e-12, 1.0e-14),
                           gamma_constant_newton = 2 * ndims(equations),
                           smoothness_indicator = false,
                           threshold_smoothness_indicator = 0.1,
                           variable_smoothness_indicator = density_pressure)
    local_twosided = (length(local_twosided_variables_cons) > 0)
    local_onesided = (length(local_onesided_variables_nonlinear) > 0)
    positivity = (length(positivity_variables_cons) +
                  length(positivity_variables_nonlinear) > 0)

    # When passing `min` or `max` in the elixir, the specific function of Base is used.
    # To speed up the simulation, we replace it with `Trixi.min` and `Trixi.max` respectively.
    local_onesided_variables_nonlinear_ = Tuple{Function, Function}[]
    for (variable, min_or_max) in local_onesided_variables_nonlinear
        if min_or_max === Base.max
            push!(local_onesided_variables_nonlinear_, (variable, max))
        elseif min_or_max === Base.min
            push!(local_onesided_variables_nonlinear_, (variable, min))
        elseif min_or_max === Trixi.max || min_or_max === Trixi.min
            push!(local_onesided_variables_nonlinear_, (variable, min_or_max))
        else
            error("Parameter $min_or_max is not a valid input. Use `max` or `min` instead.")
        end
    end
    local_onesided_variables_nonlinear_ = Tuple(local_onesided_variables_nonlinear_)

    local_twosided_variables_cons_ = get_variable_index.(local_twosided_variables_cons,
                                                         equations)
    positivity_variables_cons_ = get_variable_index.(positivity_variables_cons,
                                                     equations)

    bound_keys = ()
    if local_twosided
        for v in local_twosided_variables_cons_
            v_string = string(v)
            bound_keys = (bound_keys..., Symbol(v_string, "_min"),
                          Symbol(v_string, "_max"))
        end
    end
    if local_onesided
        for (variable, min_or_max) in local_onesided_variables_nonlinear_
            bound_keys = (bound_keys...,
                          Symbol(string(variable), "_", string(min_or_max)))
        end
    end
    for v in positivity_variables_cons_
        if !(v in local_twosided_variables_cons_)
            bound_keys = (bound_keys..., Symbol(string(v), "_min"))
        end
    end
    for variable in positivity_variables_nonlinear
        bound_keys = (bound_keys..., Symbol(string(variable), "_min"))
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
                      typeof(local_onesided_variables_nonlinear_),
                      typeof(cache),
                      typeof(IndicatorHG)}(local_twosided,
                                           local_twosided_variables_cons_,
                                           positivity, positivity_variables_cons_,
                                           positivity_variables_nonlinear,
                                           positivity_correction_factor,
                                           local_onesided,
                                           local_onesided_variables_nonlinear_,
                                           bar_states, cache,
                                           max_iterations_newton, newton_tolerances,
                                           gamma_constant_newton,
                                           smoothness_indicator,
                                           threshold_smoothness_indicator,
                                           IndicatorHG)
end

function Base.show(io::IO, limiter::SubcellLimiterIDP)
    @nospecialize limiter # reduce precompilation time
    (; local_twosided, positivity, local_onesided) = limiter

    print(io, "SubcellLimiterIDP(")
    if !(local_twosided || positivity || local_onesided)
        print(io, "No limiter selected => pure DG method")
    else
        features = String[]
        if local_twosided
            push!(features, "local min/max")
        end
        if positivity
            push!(features, "positivity")
        end
        if local_onesided
            push!(features, "local onesided")
        end
        join(io, features, ", ")
        print(io, "Limiter=($features), ")
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
    (; local_twosided, positivity, local_onesided) = limiter

    if get(io, :compact, false)
        show(io, limiter)
    else
        if !(local_twosided || positivity || local_onesided)
            setup = ["Limiter" => "No limiter selected => pure DG method"]
        else
            setup = ["Limiter" => ""]
            if local_twosided
                push!(setup,
                      "" => "Local two-sided limiting for conservative variables $(limiter.local_twosided_variables_cons)")
            end
            if positivity
                if !isempty(limiter.positivity_variables_cons)
                    string = "conservative variables $(limiter.positivity_variables_cons)"
                    push!(setup, "" => "Positivity limiting for " * string)
                end
                if !isempty(limiter.positivity_variables_nonlinear)
                    string = "$(limiter.positivity_variables_nonlinear)"
                    push!(setup, "" => "Positivity limiting for " * string)
                end
                push!(setup,
                      "" => "- with positivity correction factor = $(limiter.positivity_correction_factor)")
            end
            if local_onesided
                for (variable, min_or_max) in limiter.local_onesided_variables_nonlinear
                    push!(setup, "" => "Local $min_or_max limiting for $variable")
                end
            end
            push!(setup,
                  "Local bounds with" => (limiter.bar_states ? "Bar States" :
                                          "FV solution"))
            if limiter.smoothness_indicator
                push!(setup,
                      "Smoothness indicator" => "$(limiter.IndicatorHG) using threshold $(limiter.threshold_smoothness_indicator)")
            end
        end
        summary_box(io, "SubcellLimiterIDP", setup)
    end
end

# While for the element-wise limiting with `VolumeIntegralShockCapturingHG` the indicator is
# called here to get up-to-date values for IO, this is not easily possible in this case
# because the calculation is very integrated into the method.
# See also https://github.com/trixi-framework/Trixi.jl/pull/1611#discussion_r1334553206.
# Therefore, the coefficients at `t=t^{n-1}` are saved. Thus, the coefficients of the first
# stored solution (initial condition) are not yet defined and were manually set to `NaN`.
function get_node_variable(::Val{:limiting_coefficient}, u, mesh, equations, dg, cache)
    return dg.volume_integral.limiter.cache.subcell_limiter_coefficients.alpha
end
function get_node_variable(::Val{:limiting_coefficient}, u, mesh, equations, dg, cache,
                           equations_parabolic, cache_parabolic)
    get_node_variable(Val(:limiting_coefficient), u, mesh, equations, dg, cache)
end

###############################################################################
# Local minimum and maximum limiting (conservative variables)

@inline function idp_local_twosided!(alpha, limiter, u, t, dt, semi, elements)
    for variable in limiter.local_twosided_variables_cons
        idp_local_twosided!(alpha, limiter, u, t, dt, semi, elements, variable)
    end

    return nothing
end

##############################################################################
# Local minimum or maximum limiting (nonlinear variables)

@inline function idp_local_onesided!(alpha, limiter, u, t, dt, semi, elements)
    for (variable, min_or_max) in limiter.local_onesided_variables_nonlinear
        idp_local_onesided!(alpha, limiter, u, t, dt, semi, elements,
                            variable, min_or_max)
    end

    return nothing
end

###############################################################################
# Global positivity limiting (conservative and nonlinear variables)

@inline function idp_positivity!(alpha, limiter, u, dt, semi, elements)
    # Conservative variables
    for variable in limiter.positivity_variables_cons
        @trixi_timeit timer() "conservative variables" idp_positivity_conservative!(alpha,
                                                                                    limiter,
                                                                                    u,
                                                                                    dt,
                                                                                    semi,
                                                                                    elements,
                                                                                    variable)
    end

    # Nonlinear variables
    for variable in limiter.positivity_variables_nonlinear
        @trixi_timeit timer() "nonlinear variables" idp_positivity_nonlinear!(alpha,
                                                                              limiter,
                                                                              u,
                                                                              dt,
                                                                              semi,
                                                                              elements,
                                                                              variable)
    end

    return nothing
end

###############################################################################
# Newton-bisection method

@inline function newton_loop!(alpha, bound, u, indices, variable, min_or_max,
                              initial_check, final_check, equations, dt, limiter,
                              antidiffusive_flux)
    newton_reltol, newton_abstol = limiter.newton_tolerances

    beta = 1 - alpha[indices...]

    beta_L = 0 # alpha = 1
    beta_R = beta # No higher beta (lower alpha) than the current one

    u_curr = u + beta * dt * antidiffusive_flux

    # If state is valid, perform initial check and return if correction is not needed
    if isvalid(u_curr, equations)
        goal = goal_function_newton_idp(variable, bound, u_curr, equations)

        initial_check(min_or_max, bound, goal, newton_abstol) && return nothing
    end

    # Newton iterations
    for iter in 1:(limiter.max_iterations_newton)
        beta_old = beta

        # If the state is valid, evaluate d(goal)/d(beta)
        if isvalid(u_curr, equations)
            dgoal_dbeta = dgoal_function_newton_idp(variable, u_curr, dt,
                                                    antidiffusive_flux, equations)
        else # Otherwise, perform a bisection step
            dgoal_dbeta = 0
        end

        if dgoal_dbeta != 0
            # Update beta with Newton's method
            beta = beta - goal / dgoal_dbeta
        end

        # Check bounds
        if (beta < beta_L) || (beta > beta_R) || (dgoal_dbeta == 0) || isnan(beta)
            # Out of bounds, do a bisection step
            beta = 0.5f0 * (beta_L + beta_R)
            # Get new u
            u_curr = u + beta * dt * antidiffusive_flux

            # If the state is invalid, finish bisection step without checking tolerance and iterate further
            if !isvalid(u_curr, equations)
                beta_R = beta
                continue
            end

            # Check new beta for condition and update bounds
            goal = goal_function_newton_idp(variable, bound, u_curr, equations)
            if initial_check(min_or_max, bound, goal, newton_abstol)
                # New beta fulfills condition
                beta_L = beta
            else
                # New beta does not fulfill condition
                beta_R = beta
            end
        else
            # Get new u
            u_curr = u + beta * dt * antidiffusive_flux

            # If the state is invalid, redefine right bound without checking tolerance and iterate further
            if !isvalid(u_curr, equations)
                beta_R = beta
                continue
            end

            # Evaluate goal function
            goal = goal_function_newton_idp(variable, bound, u_curr, equations)
        end

        # Check relative tolerance
        if abs(beta_old - beta) <= newton_reltol
            break
        end

        # Check absolute tolerance
        if final_check(bound, goal, newton_abstol)
            break
        end
    end

    alpha[indices...] = 1 - beta # new alpha

    return nothing
end

### Auxiliary routines for Newton's bisection method ###
# Initial checks
@inline function initial_check_local_onesided_newton_idp(::typeof(min), bound,
                                                         goal, newton_abstol)
    return goal <= max(newton_abstol, abs(bound) * newton_abstol)
end

@inline function initial_check_local_onesided_newton_idp(::typeof(max), bound,
                                                         goal, newton_abstol)
    return goal >= -max(newton_abstol, abs(bound) * newton_abstol)
end

@inline initial_check_nonnegative_newton_idp(min_or_max, bound, goal, newton_abstol) = goal <=
                                                                                       0

# Goal and d(Goal)/d(u) function
@inline goal_function_newton_idp(variable, bound, u, equations) = bound -
                                                                  variable(u, equations)
@inline function dgoal_function_newton_idp(variable, u, dt, antidiffusive_flux,
                                           equations)
    return -dot(gradient_conservative(variable, u, equations), dt * antidiffusive_flux)
end

# Final checks
# final check for one-sided local limiting
@inline function final_check_local_onesided_newton_idp(bound, goal, newton_abstol)
    return abs(goal) < max(newton_abstol, abs(bound) * newton_abstol)
end

# final check for nonnegativity limiting
@inline function final_check_nonnegative_newton_idp(bound, goal, newton_abstol)
    return (goal <= eps()) && (goal > -max(newton_abstol, abs(bound) * newton_abstol))
end

"""
    SubcellLimiterMCL(equations::AbstractEquations, basis;
                      density_limiter = true,
                      density_coefficient_for_all = false,
                      sequential_limiter = true,
                      conservative_limiter = false,
                      positivity_limiter_pressure = false,
                      positivity_limiter_pressure_exact = true,
                      positivity_limiter_density = false,
                      positivity_limiter_correction_factor = 0.0,
                      entropy_limiter_semidiscrete = false,
                      smoothness_indicator = false,
                      threshold_smoothness_indicator = 0.1,
                      variable_smoothness_indicator = density_pressure,
                      Plotting = true)

Subcell monolithic convex limiting (MCL) used with [`VolumeIntegralSubcellLimiting`](@ref) including:
- local two-sided limiting for `cons(1)` (`density_limiter`)
- transfer amount of `density_limiter` to all quantities (`density_coefficient_for_all`)
- local two-sided limiting for variables `phi:=cons(i)/cons(1)` (`sequential_limiter`)
- local two-sided limiting for conservative variables (`conservative_limiter`)
- positivity limiting for `cons(1)` (`positivity_limiter_density`)
- positivity limiting pressure à la Kuzmin (`positivity_limiter_pressure`)
- semidiscrete entropy fix (`entropy_limiter_semidiscrete`)

The pressure positivity limiting preserves a sharp version (`positivity_limiter_pressure_exact`)
and a more cautious one. The density positivity limiter uses a `positivity_limiter_correction_factor`
such that `u^new >= positivity_limiter_correction_factor * u^FV`. All additional analyses for plotting
routines can be disabled via `Plotting=false` (see `save_alpha` and `update_alpha_max_avg!`).

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
    density_limiter::Bool               # Impose local maximum/minimum for cons(1) based on bar states
    density_coefficient_for_all::Bool   # Use the cons(1) blending coefficient for all quantities
    sequential_limiter::Bool    # Impose local maximum/minimum for variables phi:=cons(i)/cons(1) i 2:nvariables based on bar states
    conservative_limiter::Bool  # Impose local maximum/minimum for conservative variables 2:nvariables based on bar states
    positivity_limiter_pressure::Bool       # Impose positivity for pressure  la Kuzmin
    positivity_limiter_pressure_exact::Bool # Only for positivity_limiter_pressure=true: Use the sharp calculation of factor
    positivity_limiter_density::Bool        # Impose positivity for cons(1)
    positivity_limiter_correction_factor::RealT  # Correction Factor for positivity_limiter_density in [0,1)
    entropy_limiter_semidiscrete::Bool      # synchronized semidiscrete entropy fix
    smoothness_indicator::Bool              # activates smoothness indicator: IndicatorHennemannGassner
    threshold_smoothness_indicator::RealT   # threshold for smoothness indicator
    IndicatorHG::Indicator
    Plotting::Bool
end

# this method is used when the limiter is constructed as for shock-capturing volume integrals
function SubcellLimiterMCL(equations::AbstractEquations, basis;
                           density_limiter = true,
                           density_coefficient_for_all = false,
                           sequential_limiter = true,
                           conservative_limiter = false,
                           positivity_limiter_pressure = false,
                           positivity_limiter_pressure_exact = true,
                           positivity_limiter_density = false,
                           positivity_limiter_correction_factor = 0.0,
                           entropy_limiter_semidiscrete = false,
                           smoothness_indicator = false,
                           threshold_smoothness_indicator = 0.1,
                           variable_smoothness_indicator = density_pressure,
                           Plotting = true)
    if sequential_limiter && conservative_limiter
        error("Only one of the two can be selected: sequential_limiter/conservative_limiter")
    end
    cache = create_cache(SubcellLimiterMCL, equations, basis,
                         positivity_limiter_pressure)
    if smoothness_indicator
        IndicatorHG = IndicatorHennemannGassner(equations, basis, alpha_smooth = false,
                                                variable = variable_smoothness_indicator)
    else
        IndicatorHG = nothing
    end
    SubcellLimiterMCL{typeof(threshold_smoothness_indicator), typeof(cache),
                      typeof(IndicatorHG)}(cache,
                                           density_limiter, density_coefficient_for_all,
                                           sequential_limiter, conservative_limiter,
                                           positivity_limiter_pressure,
                                           positivity_limiter_pressure_exact,
                                           positivity_limiter_density,
                                           positivity_limiter_correction_factor,
                                           entropy_limiter_semidiscrete,
                                           smoothness_indicator,
                                           threshold_smoothness_indicator, IndicatorHG,
                                           Plotting)
end

function Base.show(io::IO, limiter::SubcellLimiterMCL)
    @nospecialize limiter # reduce precompilation time

    print(io, "SubcellLimiterMCL(")
    limiter.density_limiter && print(io, "; dens")
    limiter.density_coefficient_for_all && print(io, "; dens alpha ∀")
    limiter.sequential_limiter && print(io, "; seq")
    limiter.conservative_limiter && print(io, "; cons")
    if limiter.positivity_limiter_pressure
        print(io,
              "; $(limiter.positivity_limiter_pressure_exact ? "pres (sharp)" : "pres (cautious)")")
    end
    limiter.positivity_limiter_density && print(io, "; dens pos")
    if limiter.positivity_limiter_correction_factor != 0
        print(io,
              " with correction factor $(limiter.positivity_limiter_correction_factor)")
    end
    limiter.entropy_limiter_semidiscrete && print(io, "; semid. entropy")
    limiter.smoothness_indicator &&
        print(io, "; Smoothness indicator: ", limiter.IndicatorHG,
              " with threshold ", limiter.threshold_smoothness_indicator)
    print(io, ")")
end

function Base.show(io::IO, ::MIME"text/plain", limiter::SubcellLimiterMCL)
    @nospecialize limiter # reduce precompilation time
    @unpack density_limiter, density_coefficient_for_all, sequential_limiter, conservative_limiter,
    positivity_limiter_pressure_exact, positivity_limiter_density, entropy_limiter_semidiscrete = limiter

    if get(io, :compact, false)
        show(io, limiter)
    else
        setup = ["limiter" => ""]
        density_limiter && (setup = [setup..., "" => "Density Limiter"])
        density_coefficient_for_all &&
            (setup = [setup..., "" => "Transfer density coefficient to all quantities"])
        sequential_limiter && (setup = [setup..., "" => "Sequential Limiter"])
        conservative_limiter && (setup = [setup..., "" => "Conservative Limiter"])
        if limiter.positivity_limiter_pressure
            setup = [
                setup...,
                "" => "$(positivity_limiter_pressure_exact ? "(Sharp)" : "(Cautious)") positivity limiter for Pressure à la Kuzmin"
            ]
        end
        if positivity_limiter_density
            if limiter.positivity_limiter_correction_factor != 0.0
                setup = [
                    setup...,
                    "" => "Positivity Limiter for Density with correction factor $(limiter.positivity_limiter_correction_factor)"
                ]
            else
                setup = [setup..., "" => "Positivity Limiter for Density"]
            end
        end
        entropy_limiter_semidiscrete &&
            (setup = [setup..., "" => "Semidiscrete Entropy Limiter"])
        if limiter.smoothness_indicator
            setup = [
                setup...,
                "Smoothness indicator" => "$(limiter.IndicatorHG) using threshold $(limiter.threshold_smoothness_indicator)"
            ]
        end
        summary_box(io, "SubcellLimiterMCL", setup)
    end
end

function get_node_variable(::Val{:limiting_coefficient_rho}, u,
                           mesh, equations, dg, cache)
    (; limiter) = dg.volume_integral
    if !limiter.Plotting
        error("Activate `limiter.Plotting` to allow saving of limiting coefficients for MCL.")
    end
    (; alpha) = limiter.cache.subcell_limiter_coefficients
    return alpha[1, ntuple(_ -> :, size(alpha, 2) + 1)...]
end

function get_node_variable(::Val{:limiting_coefficient_rho_v1}, u,
                           mesh, equations, dg, cache)
    (; limiter) = dg.volume_integral
    if !limiter.Plotting
        error("Activate `limiter.Plotting` to allow saving of limiting coefficients for MCL.")
    end
    (; alpha) = limiter.cache.subcell_limiter_coefficients
    return alpha[2, ntuple(_ -> :, size(alpha, 2) + 1)...]
end

function get_node_variable(::Val{:limiting_coefficient_rho_v2}, u,
                           mesh, equations, dg, cache)
    (; limiter) = dg.volume_integral
    if !limiter.Plotting
        error("Activate `limiter.Plotting` to allow saving of limiting coefficients for MCL.")
    end
    (; alpha) = limiter.cache.subcell_limiter_coefficients
    return alpha[3, ntuple(_ -> :, size(alpha, 2) + 1)...]
end

function get_node_variable(::Val{:limiting_coefficient_rho_e}, u,
                           mesh, equations, dg, cache)
    (; limiter) = dg.volume_integral
    if !limiter.Plotting
        error("Activate `limiter.Plotting` to allow saving of limiting coefficients for MCL.")
    end
    (; alpha) = limiter.cache.subcell_limiter_coefficients
    return alpha[4, ntuple(_ -> :, size(alpha, 2) + 1)...]
end

function get_node_variable(::Val{:limiting_coefficient_pressure}, u,
                           mesh, equations, dg, cache)
    (; limiter) = dg.volume_integral
    if !limiter.Plotting
        error("Activate `limiter.Plotting` to allow saving of limiting coefficients for MCL.")
    end
    (; alpha_pressure) = limiter.cache.subcell_limiter_coefficients
    return alpha_pressure
end

function get_node_variable(::Val{:limiting_coefficient_entropy}, u,
                           mesh, equations, dg, cache)
    (; limiter) = dg.volume_integral
    if !limiter.Plotting
        error("Activate `limiter.Plotting` to allow saving of limiting coefficients for MCL.")
    end
    (; alpha_entropy) = limiter.cache.subcell_limiter_coefficients
    return alpha_entropy
end

function get_node_variable(::Val{:limiting_coefficient_mean_rho}, u,
                           mesh, equations, dg, cache)
    (; limiter) = dg.volume_integral
    if !limiter.Plotting
        error("Activate `limiter.Plotting` to allow saving of limiting coefficients for MCL.")
    end
    (; alpha_mean) = limiter.cache.subcell_limiter_coefficients
    return alpha_mean[1, ntuple(_ -> :, size(alpha_mean, 2) + 1)...]
end

function get_node_variable(::Val{:limiting_coefficient_mean_rho_v1}, u,
                           mesh, equations, dg, cache)
    (; limiter) = dg.volume_integral
    if !limiter.Plotting
        error("Activate `limiter.Plotting` to allow saving of limiting coefficients for MCL.")
    end
    (; alpha_mean) = limiter.cache.subcell_limiter_coefficients
    return alpha_mean[2, ntuple(_ -> :, size(alpha_mean, 2) + 1)...]
end

function get_node_variable(::Val{:limiting_coefficient_mean_rho_v2}, u,
                           mesh, equations, dg, cache)
    (; limiter) = dg.volume_integral
    if !limiter.Plotting
        error("Activate `limiter.Plotting` to allow saving of limiting coefficients for MCL.")
    end
    (; alpha_mean) = limiter.cache.subcell_limiter_coefficients
    return alpha_mean[3, ntuple(_ -> :, size(alpha_mean, 2) + 1)...]
end

function get_node_variable(::Val{:limiting_coefficient_mean_rho_e}, u,
                           mesh, equations, dg, cache)
    (; limiter) = dg.volume_integral
    if !limiter.Plotting
        error("Activate `limiter.Plotting` to allow saving of limiting coefficients for MCL.")
    end
    (; alpha_mean) = limiter.cache.subcell_limiter_coefficients
    return alpha_mean[4, ntuple(_ -> :, size(alpha_mean, 2) + 1)...]
end

function get_node_variable(::Val{:limiting_coefficient_mean_pressure}, u,
                           mesh, equations, dg, cache)
    (; limiter) = dg.volume_integral
    if !limiter.Plotting
        error("Activate `limiter.Plotting` to allow saving of limiting coefficients for MCL.")
    end
    (; alpha_mean_pressure) = limiter.cache.subcell_limiter_coefficients
    return alpha_mean_pressure
end

function get_node_variable(::Val{:limiting_coefficient_mean_entropy}, u,
                           mesh, equations, dg, cache)
    (; limiter) = dg.volume_integral
    if !limiter.Plotting
        error("Activate `limiter.Plotting` to allow saving of limiting coefficients for MCL.")
    end
    (; alpha_mean_entropy) = limiter.cache.subcell_limiter_coefficients
    return alpha_mean_entropy
end
end # @muladd
