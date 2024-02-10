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

"""
    SubcellLimiterIDP(equations::AbstractEquations, basis;
                      local_minmax_variables_cons = String[],
                      positivity_variables_cons = String[],
                      positivity_variables_nonlinear = [],
                      positivity_correction_factor = 0.1,
                      max_iterations_newton = 10,
                      newton_tolerances = (1.0e-12, 1.0e-14),
                      gamma_constant_newton = 2 * ndims(equations))

Subcell invariant domain preserving (IDP) limiting used with [`VolumeIntegralSubcellLimiting`](@ref)
including:
- Local maximum/minimum Zalesak-type limiting for conservative variables (`local_minmax_variables_cons`)
- Positivity limiting for conservative variables (`positivity_variables_cons`) and nonlinear variables
(`positivity_variables_nonlinear`)

Conservative variables to be limited are passed as a vector of strings, e.g. `local_minmax_variables_cons = ["rho"]`
and `positivity_variables_cons = ["rho"]`. For nonlinear variables the specific functions are
passed in a vector, e.g. `positivity_variables_nonlinear = [pressure]`.

The bounds are calculated using the low-order FV solution. The positivity limiter uses
`positivity_correction_factor` such that `u^new >= positivity_correction_factor * u^FV`.
The limiting of nonlinear variables uses a Newton-bisection method with a maximum of
`max_iterations_newton` iterations, relative and absolute tolerances of `newton_tolerances`
and a provisional update constant `gamma_constant_newton` (`gamma_constant_newton>=2*d`,
where `d = #dimensions`). See equation (20) of Pazner (2020) and equation (30) of Rueda-Ramírez et al. (2022).

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

!!! warning "Experimental implementation"
    This is an experimental feature and may change in future releases.
"""
struct SubcellLimiterIDP{RealT <: Real, LimitingVariablesNonlinear, Cache} <:
       AbstractSubcellLimiter
    local_minmax::Bool
    local_minmax_variables_cons::Vector{Int}                   # Local mininum/maximum principles for conservative variables
    positivity::Bool
    positivity_variables_cons::Vector{Int}                     # Positivity for conservative variables
    positivity_variables_nonlinear::LimitingVariablesNonlinear # Positivity for nonlinear variables
    positivity_correction_factor::RealT
    cache::Cache
    max_iterations_newton::Int
    newton_tolerances::Tuple{RealT, RealT}  # Relative and absolute tolerances for Newton's method
    gamma_constant_newton::RealT            # Constant for the subcell limiting of convex (nonlinear) constraints
end

# this method is used when the limiter is constructed as for shock-capturing volume integrals
function SubcellLimiterIDP(equations::AbstractEquations, basis;
                           local_minmax_variables_cons = String[],
                           positivity_variables_cons = String[],
                           positivity_variables_nonlinear = [],
                           positivity_correction_factor = 0.1,
                           max_iterations_newton = 10,
                           newton_tolerances = (1.0e-12, 1.0e-14),
                           gamma_constant_newton = 2 * ndims(equations))
    local_minmax = (length(local_minmax_variables_cons) > 0)
    positivity = (length(positivity_variables_cons) +
                  length(positivity_variables_nonlinear) > 0)

    local_minmax_variables_cons_ = get_variable_index.(local_minmax_variables_cons,
                                                       equations)
    positivity_variables_cons_ = get_variable_index.(positivity_variables_cons,
                                                     equations)

    bound_keys = ()
    if local_minmax
        for v in local_minmax_variables_cons_
            v_string = string(v)
            bound_keys = (bound_keys..., Symbol(v_string, "_min"),
                          Symbol(v_string, "_max"))
        end
    end
    for v in positivity_variables_cons_
        if !(v in local_minmax_variables_cons_)
            bound_keys = (bound_keys..., Symbol(string(v), "_min"))
        end
    end
    for variable in positivity_variables_nonlinear
        bound_keys = (bound_keys..., Symbol(string(variable), "_min"))
    end

    cache = create_cache(SubcellLimiterIDP, equations, basis, bound_keys)

    SubcellLimiterIDP{typeof(positivity_correction_factor),
                      typeof(positivity_variables_nonlinear),
                      typeof(cache)}(local_minmax, local_minmax_variables_cons_,
                                     positivity, positivity_variables_cons_,
                                     positivity_variables_nonlinear,
                                     positivity_correction_factor, cache,
                                     max_iterations_newton, newton_tolerances,
                                     gamma_constant_newton)
end

function Base.show(io::IO, limiter::SubcellLimiterIDP)
    @nospecialize limiter # reduce precompilation time
    (; local_minmax, positivity) = limiter

    print(io, "SubcellLimiterIDP(")
    if !(local_minmax || positivity)
        print(io, "No limiter selected => pure DG method")
    else
        features = String[]
        if local_minmax
            push!(features, "local min/max")
        end
        if positivity
            push!(features, "positivity")
        end
        join(io, features, ", ")
        print(io, "Limiter=($features), ")
    end
    print(io, "Local bounds with FV solution")
    print(io, ")")
end

function Base.show(io::IO, ::MIME"text/plain", limiter::SubcellLimiterIDP)
    @nospecialize limiter # reduce precompilation time
    (; local_minmax, positivity) = limiter

    if get(io, :compact, false)
        show(io, limiter)
    else
        if !(local_minmax || positivity)
            setup = ["limiter" => "No limiter selected => pure DG method"]
        else
            setup = ["limiter" => ""]
            if local_minmax
                setup = [setup...,
                         "" => "Local maximum/minimum limiting for conservative variables $(limiter.local_minmax_variables_cons)"]
            end
            if positivity
                string = "Positivity limiting for conservative variables $(limiter.positivity_variables_cons) and $(limiter.positivity_variables_nonlinear)"
                setup = [setup..., "" => string]
                setup = [setup...,
                         "" => "- with positivity correction factor = $(limiter.positivity_correction_factor)"]
            end
            setup = [setup...,
                     "Local bounds" => "FV solution"]
        end
        summary_box(io, "SubcellLimiterIDP", setup)
    end
end

function get_node_variables!(node_variables, limiter::SubcellLimiterIDP,
                             ::VolumeIntegralSubcellLimiting, equations)
    node_variables[:limiting_coefficient] = limiter.cache.subcell_limiter_coefficients.alpha

    return nothing
end
end # @muladd
