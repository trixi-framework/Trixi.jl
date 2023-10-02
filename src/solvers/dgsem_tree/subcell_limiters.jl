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
                      positivity_variables_cons = [],
                      positivity_correction_factor = 0.1)

Subcell invariant domain preserving (IDP) limiting used with [`VolumeIntegralSubcellLimiting`](@ref)
including:
- positivity limiting for conservative variables (`positivity_variables_cons`)

The bounds are calculated using the low-order FV solution. The positivity limiter uses
`positivity_correction_factor` such that `u^new >= positivity_correction_factor * u^FV`.

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
struct SubcellLimiterIDP{RealT <: Real, Cache} <: AbstractSubcellLimiter
    positivity::Bool
    positivity_variables_cons::Vector{Int}                     # Positivity for conservative variables
    positivity_correction_factor::RealT
    cache::Cache
end

# this method is used when the indicator is constructed as for shock-capturing volume integrals
function SubcellLimiterIDP(equations::AbstractEquations, basis;
                           positivity_variables_cons = [],
                           positivity_correction_factor = 0.1)
    positivity = (length(positivity_variables_cons) > 0)

    bound_keys = Tuple(Symbol("$(i)_min") for i in positivity_variables_cons)

    cache = create_cache(SubcellLimiterIDP, equations, basis, bound_keys)

    SubcellLimiterIDP{typeof(positivity_correction_factor), typeof(cache)}(positivity,
                                                                           positivity_variables_cons,
                                                                           positivity_correction_factor,
                                                                           cache)
end

function Base.show(io::IO, limiter::SubcellLimiterIDP)
    @nospecialize limiter # reduce precompilation time
    @unpack positivity = limiter

    print(io, "SubcellLimiterIDP(")
    if !(positivity)
        print(io, "No limiter selected => pure DG method")
    else
        print(io, "limiter=(")
        positivity && print(io, "positivity")
        print(io, "), ")
    end
    print(io, ")")
end

function Base.show(io::IO, ::MIME"text/plain", limiter::SubcellLimiterIDP)
    @nospecialize limiter # reduce precompilation time
    @unpack positivity = limiter

    if get(io, :compact, false)
        show(io, limiter)
    else
        if !(positivity)
            setup = ["limiter" => "No limiter selected => pure DG method"]
        else
            setup = ["limiter" => ""]
            if positivity
                string = "positivity with conservative variables $(limiter.positivity_variables_cons)"
                setup = [setup..., "" => string]
                setup = [
                    setup...,
                    "" => "   positivity correction factor = $(limiter.positivity_correction_factor)",
                ]
            end
        end
        summary_box(io, "SubcellLimiterIDP", setup)
    end
end
end # @muladd
