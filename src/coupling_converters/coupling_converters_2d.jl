# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

@doc raw"""
    Coupling converter function for a system of two LinearScalarAdvectionEquation2D.

The coupling is given as a Heaviside step.
```math
c(x) = {c_0, for x \ge x_0 \times s
        0, for x < x_0 \times s}
```
Here, `s` is the sign of the step function, x_0 the save_position
of the step and c_0 the amplitude.
"""
function coupling_converter_heaviside_2d(x, x_0, c_0, s,
                                         equations_left::LinearScalarAdvectionEquation2D,
                                         equation_right::LinearScalarAdvectionEquation2D)
    return c_0 * (s*sign(x[2] - x_0) + 1.0)/2.0
end

end # @muladd
