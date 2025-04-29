# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

# This file contains analysis computations that are performed on the surface, 
# such as aerodynamic coefficients.

"""
    AnalysisSurfaceIntegral{Variable, NBoundaries}(boundary_symbols::NTuple{NBoundaries, Symbol},
                                                   variable)

This struct is used to compute the surface integral of a quantity of interest `variable` alongside
the boundary/boundaries associated with particular names given in `boundary_symbols`.
For instance, this can be used to compute the lift [`LiftCoefficientPressure2D`](@ref) or
drag coefficient [`DragCoefficientPressure2D`](@ref) of e.g. an 2D airfoil with the boundary
names `:AirfoilTop`, `:AirfoilBottom` which would be supplied as 
`boundary_symbols = (:AirfoilTop, :AirfoilBottom)`.
A single boundary name can also be supplied, e.g. `boundary_symbols = (:AirfoilTop,)`.

- `boundary_symbols::NTuple{NBoundaries, Symbol}`: Name(s) of the boundary/boundaries
  where the quantity of interest is computed
- `variable::Variable`: Quantity of interest, like lift or drag
"""
struct AnalysisSurfaceIntegral{Variable, NBoundaries}
    variable::Variable # Quantity of interest, like lift or drag
    boundary_symbols::NTuple{NBoundaries, Symbol} # Name(s) of the boundary/boundaries

    function AnalysisSurfaceIntegral(boundary_symbols::NTuple{NBoundaries, Symbol},
                                     variable) where {NBoundaries}
        return new{typeof(variable), NBoundaries}(variable, boundary_symbols)
    end
end

# This returns the boundary indices of a given iterable datastructure of boundary symbols.
function get_boundary_indices(boundary_symbols, boundary_symbol_indices)
    indices = Int[]
    for name in boundary_symbols
        append!(indices, boundary_symbol_indices[name])
    end
    sort!(indices) # Try to achieve some data locality by sorting

    return indices
end

struct ForceState{RealT <: Real, NDIMS}
    psi::NTuple{NDIMS, RealT} # Unit vector normal or parallel to freestream
    rho_inf::RealT
    u_inf::RealT
    l_inf::RealT
end

# Abstract base type used for dispatch of `analyze` for quantities
# requiring gradients of the velocity field.
abstract type VariableViscous end

struct LiftCoefficientPressure{RealT <: Real, NDIMS}
    force_state::ForceState{RealT, NDIMS}
end

struct DragCoefficientPressure{RealT <: Real, NDIMS}
    force_state::ForceState{RealT, NDIMS}
end

struct LiftCoefficientShearStress{RealT <: Real, NDIMS} <: VariableViscous
    force_state::ForceState{RealT, NDIMS}
end

struct DragCoefficientShearStress{RealT <: Real, NDIMS} <: VariableViscous
    force_state::ForceState{RealT, NDIMS}
end

function (lift_coefficient::LiftCoefficientPressure)(u, normal_direction, x, t,
                                                     equations)
    p = pressure(u, equations)
    @unpack psi, rho_inf, u_inf, l_inf = lift_coefficient.force_state
    # Normalize as `normal_direction` is not necessarily a unit vector
    n = dot(normal_direction, psi) / norm(normal_direction)
    return p * n / (0.5 * rho_inf * u_inf^2 * l_inf)
end

function (drag_coefficient::DragCoefficientPressure)(u, normal_direction, x, t,
                                                     equations)
    p = pressure(u, equations)
    @unpack psi, rho_inf, u_inf, l_inf = drag_coefficient.force_state
    # Normalize as `normal_direction` is not necessarily a unit vector
    n = dot(normal_direction, psi) / norm(normal_direction)
    return p * n / (0.5 * rho_inf * u_inf^2 * l_inf)
end

function pretty_form_ascii(::AnalysisSurfaceIntegral{<:LiftCoefficientPressure{<:Any,
                                                                               <:Any}})
    "CL_p"
end
function pretty_form_utf(::AnalysisSurfaceIntegral{<:LiftCoefficientPressure{<:Any,
                                                                             <:Any}})
    "CL_p"
end

function pretty_form_ascii(::AnalysisSurfaceIntegral{<:DragCoefficientPressure{<:Any,
                                                                               <:Any}})
    "CD_p"
end
function pretty_form_utf(::AnalysisSurfaceIntegral{<:DragCoefficientPressure{<:Any,
                                                                             <:Any}})
    "CD_p"
end

function pretty_form_ascii(::AnalysisSurfaceIntegral{<:LiftCoefficientShearStress{<:Any,
                                                                                  <:Any}})
    "CL_f"
end
function pretty_form_utf(::AnalysisSurfaceIntegral{<:LiftCoefficientShearStress{<:Any,
                                                                                <:Any}})
    "CL_f"
end

function pretty_form_ascii(::AnalysisSurfaceIntegral{<:DragCoefficientShearStress{<:Any,
                                                                                  <:Any}})
    "CD_f"
end
function pretty_form_utf(::AnalysisSurfaceIntegral{<:DragCoefficientShearStress{<:Any,
                                                                                <:Any}})
    "CD_f"
end

include("analysis_surface_integral_2d.jl")
end # muladd
