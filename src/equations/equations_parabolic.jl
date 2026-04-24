# specify transformation of conservative variables prior to taking gradients.
# specialize this function to compute gradients e.g., of primitive variables instead of conservative
gradient_variable_transformation(::AbstractEquationsParabolic) = cons2cons

# By default, the gradients are taken with respect to the conservative variables.
# this is reflected by the type parameter `GradientVariablesConservative` in the abstract
# type `AbstractEquationsParabolic{NDIMS, NVARS, GradientVariablesConservative}`.
struct GradientVariablesConservative end

include("laplace_diffusion.jl")

include("laplace_diffusion_entropy_variables.jl")

include("linear_diffusion_equation.jl")

include("compressible_navier_stokes.jl")

include("visco_resistive_mhd_2d.jl")
