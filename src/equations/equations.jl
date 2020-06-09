
# Base type from which all systems of equations types inherit from
abstract type AbstractEquation{V} end


# Retrieve number of variables from equation type
nvariables(::Type{AbstractEquation{V}}) where V = V

# Retrieve number of variables from equation instance
nvariables(::AbstractEquation{V}) where V = V


# Add method to show some information on system of equations
function Base.show(io::IO, equation::AbstractEquation)
  print(io, "name = ", get_name(equation), ", n_vars = ", nvariables(equation))
end


# Create an instance of a system of equation type based on a given name
function make_equations(name::String)
  if name == "LinearScalarAdvectionEquation"
    return LinearScalarAdvectionEquation()
  elseif name == "CompressibleEulerEquations"
    return CompressibleEulerEquations()
  elseif name == "IdealGlmMhdEquations"
    return IdealGlmMhdEquations()
  elseif name == "HyperbolicDiffusionEquations"
    return HyperbolicDiffusionEquations()
  else
    error("'$name' does not name a valid system of equations")
  end
end


have_nonconservative_terms(::AbstractEquation) = Val(false)
default_analysis_quantities(::AbstractEquation) = (:l2_error, :linf_error, :dsdu_ut)


"""
    riemann!(destination, surface_flux, u_surfaces_left, u_surfaces_right, surface_id,
             equation::AbstractEquation, n_nodes, orientations)

Calculate the `surface_flux` across interface with different states given by
`u_surfaces_left, u_surfaces_right` on both sides (EC mortar version).

# Arguments
- `destination::AbstractArray{T,3} where T<:Real`:
  The array of surface flux values (updated inplace).
- `surface_flux`:
  The surface flux as a function.
- `u_surfaces_left::AbstractArray{T,3} where T<:Real``
- `u_surfaces_right::AbstractArray{T,3} where T<:Real``
- `surface_id::Integer`
- `equation::AbstractEquations`
- `n_nodes::Integer`
- `orientations::Vector{T} where T<:Integer`
"""
function riemann!(destination, surface_flux, u_surfaces_left, u_surfaces_right, surface_id,
                  equation::AbstractEquation, n_nodes, orientations) end

"""
    riemann!(destination, surface_flux, u_surfaces, surface_id,
             equation::AbstractEquation, n_nodes, orientations)

Calculate the `surface_flux` across interface with different states given by
`u_surfaces_left, u_surfaces_right` on both sides (surface version).

# Arguments
- `destination::AbstractArray{T,2} where T<:Real`:
  The array of surface flux values (updated inplace).
- `surface_flux`:
  The surface flux as a function.
- `u_surfaces::AbstractArray{T,4} where T<:Real``
- `surface_id::Integer`
- `equation::AbstractEquations`
- `n_nodes::Integer`
- `orientations::Vector{T} where T<:Integer`
"""
function riemann!(destination, surface_flux, u_surfaces, surface_id,
                  equation::AbstractEquation, n_nodes, orientations) end


####################################################################################################
# Include files with actual implementations for different systems of equations.

# Linear scalar advection
include("linear_scalar_advection.jl")

# CompressibleEulerEquations
include("compressible_euler.jl")

# Ideal MHD
include("ideal_glm_mhd.jl")

# Diffusion equation: first order hyperbolic system
include("hyperbolic_diffusion.jl")
