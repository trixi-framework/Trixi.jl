# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

@doc raw"""
    Identity coupling converter function.

The coupling is given as a linear function.
```math
c(x) = u(x)
```
"""
function coupling_converter_identity(equations::AbstractEquations)
    return (x, u) -> u
end

####################################################################################################
# Include files with actual implementations for different systems of equations.

include("coupling_converters_2d.jl")
end # @muladd
