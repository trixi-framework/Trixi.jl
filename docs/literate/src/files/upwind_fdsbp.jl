#src # Upwind FD SBP schemes

# General tensor product SBP methods are supported via the `DGMulti` solver
# in a reasonably complete way, see [the previous tutorial](@ref DGMulti_2).
# Nevertheless, there is also experimental support for SBP methods with
# other solver and mesh types.

# The first step is to set up an SBP operator. A classical (central) SBP
# operator can be created as follows.
D_SBP = derivative_operator(SummationByPartsOperators.MattssonNordström2004(),
                            derivative_order=1, accuracy_order=2,
                            xmin=0.0, xmax=1.0, N=11)
# Instead of prefixing the source of coefficients `MattssonNordström2004()`,
# you can also load the package SummationByPartsOperators.jl.
# This yields an object representing the operator efficiently. If you want to
# compare it to coefficients presented in the literature, you can convert it
# to a matrix.
Matrix(D_SBP)

# Upwind SBP operators are a concept introduced in 2017 by Ken Mattsson. You can
# create them as follows.
D_upw = upwind_operators(SummationByPartsOperators.Mattsson2017,
                         derivative_order=1, accuracy_order=2,
                         xmin=0.0, xmax=1.0, N=11)
# Upwind operators are derivative operators biased towards one direction.
# The "minus" variants has a bias towards the left side.
Matrix(D_upw.minus)
# Analogously, the "plus" variant has a bias towards the right side.
Matrix(D_upw.plus)
# For more information on upwind SBP operators, please refer to the documentation
# of [SummationByPartsOperators.jl](https://github.com/ranocha/SummationByPartsOperators.jl)
# and references cited there.
