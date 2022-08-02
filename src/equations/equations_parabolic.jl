# specify transformation of conservative variables prior to taking gradients.
# specialize this function to compute gradients e.g., of primitive variables instead of conservative
gradient_variable_transformation(::AbstractEquationsParabolic, dg_parabolic) = cons2cons

# Linear scalar diffusion for use in linear scalar advection-diffusion problems
abstract type AbstractLaplaceDiffusionEquations{NDIMS, NVARS} <: AbstractEquationsParabolic{NDIMS, NVARS} end
include("laplace_diffusion_2d.jl")

# Compressible Navier-Stokes equations
abstract type AbstractCompressibleNavierStokesEquations{NDIMS, NVARS} <: AbstractEquationsParabolic{NDIMS, NVARS} end
include("compressible_navier_stokes_2d.jl")

"""
  EquationsHyperbolicParabolic(equations_hyperbolic, equations_parabolic)

A type representing a combined hyperbolic-parabolic equation. Expects that both
equations have the same number of variables.
"""
struct EquationsHyperbolicParabolic{H<:AbstractEquations, P<:AbstractEquationsParabolic, NDIMS, NVARS} <: AbstractEquations{NDIMS, NVARS}
  equations_hyperbolic::H
  equations_parabolic::P

  function EquationsHyperbolicParabolic(equations_hyperbolic, equations_parabolic)
    @assert nvariables(equations_hyperbolic) == nvariables(equations_parabolic)
    return new{typeof(equations_hyperbolic), typeof(equations_parabolic), ndims(equations_hyperbolic), nvariables(equations_hyperbolic)}(equations_hyperbolic, equations_parabolic)
  end
end

# default to using nvariables and varnames of hyperbolic equations
varnames(variable_mapping, equations::EquationsHyperbolicParabolic) =
  varnames(variable_mapping, equations.equations_hyperbolic)

nvariables(equations::EquationsHyperbolicParabolic) =
  nvariables(equations.equations_hyperbolic)

function Base.show(io::IO, text_type::MIME"text/plain", equations::EquationsHyperbolicParabolic)
  # Since this is not performance-critical, we can use `@nospecialize` to reduce latency.
  @nospecialize equations # reduce precompilation time

  if get(io, :compact, false)
    show(io, equations)
  else
    summary_header(io, get_name(equations) *
      "{" * get_name(equations.equations_hyperbolic) * ", "
          * get_name(equations.equations_parabolic) * "}")
    summary_interior(io, equations.equations_hyperbolic)
    summary_footer(io)
  end
end

function summary_interior(io, equations)
  summary_line(io, "#variables", nvariables(equations))
  for variable in eachvariable(equations)
    summary_line(increment_indent(io),
                 "variable " * string(variable),
                 varnames(cons2cons, equations)[variable])
  end
end
