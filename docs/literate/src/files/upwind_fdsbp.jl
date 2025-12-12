#src # Upwind FD SBP schemes

# General tensor product SBP methods are supported via the `DGMulti` solver
# in a reasonably complete way, see [the previous tutorial](@ref DGMulti_2).
# Nevertheless, there is also experimental support for SBP methods with
# other solver and mesh types.

# The first step is to set up an SBP operator. A classical (central) SBP
# operator can be created as follows.
using Trixi
D_SBP = derivative_operator(SummationByPartsOperators.MattssonNordström2004(),
                            derivative_order = 1, accuracy_order = 2,
                            xmin = 0.0, xmax = 1.0, N = 11)
# Instead of prefixing the source of coefficients `MattssonNordström2004()`,
# you can also load the package SummationByPartsOperators.jl. Either way,
# this yields an object representing the operator efficiently. If you want to
# compare it to coefficients presented in the literature, you can convert it
# to a matrix.
Matrix(D_SBP)

# Upwind SBP operators are a concept introduced in 2017 by Ken Mattsson. You can
# create them as follows.
D_upw = upwind_operators(SummationByPartsOperators.Mattsson2017,
                         derivative_order = 1, accuracy_order = 2,
                         xmin = 0.0, xmax = 1.0, N = 11)
# Upwind operators are derivative operators biased towards one direction.
# The "minus" variants has a bias towards the left side, i.e., it uses values
# from more nodes to the left than from the right to compute the discrete
# derivative approximation at a given node (in the interior of the domain).
# In matrix form, this means more non-zero entries are left from the diagonal.
Matrix(D_upw.minus)
# Analogously, the "plus" variant has a bias towards the right side.
Matrix(D_upw.plus)
# For more information on upwind SBP operators, please refer to the documentation
# of [SummationByPartsOperators.jl](https://github.com/ranocha/SummationByPartsOperators.jl)
# and references cited there.

# The basic idea of upwind SBP schemes is to apply a flux vector splitting and
# use appropriate upwind operators for both parts of the flux. In 1D, this means
# to split the flux
# ```math
# f(u) = f^-(u) + f^+(u)
# ```
# such that $f^-(u)$ is associated with left-going waves and $f^+(u)$ with
# right-going waves. Then, we apply upwind SBP operators $D^-, D^+$ with an
# appropriate upwind bias, resulting in
# ```math
# \partial_x f(u) \approx D^+ f^-(u) + D^- f^+(u)
# ```
# Note that the established notations of upwind operators $D^\pm$ and flux
# splittings $f^\pm$ clash. The right-going waves from $f^+$ need an operator
# biased towards their upwind side, i.e., the left side. This upwind bias is
# provided by the operator $D^-$.

# Many classical flux vector splittings have been developed for finite volume
# methods and are described in the book "Riemann Solvers and Numerical Methods
# for Fluid Dynamics: A Practical Introduction" of Eleuterio F. Toro (2009),
# [DOI: 10.1007/b79761](https://doi.org/10.1007/b79761). One such a well-known
# splitting provided by Trixi.jl is [`splitting_steger_warming`](@ref).

# Trixi.jl comes with several example setups using upwind SBP methods with
# flux vector splitting, e.g.,
# - [`elixir_euler_vortex.jl`](https://github.com/trixi-framework/Trixi.jl/blob/main/examples/tree_2d_fdsbp/elixir_euler_vortex.jl)
# - [`elixir_euler_taylor_green_vortex.jl`](https://github.com/trixi-framework/Trixi.jl/blob/main/examples/tree_3d_fdsbp/elixir_euler_taylor_green_vortex.jl)

# ## Package versions

# These results were obtained using the following versions.

using InteractiveUtils
versioninfo()

using Pkg
Pkg.status(["Trixi", "SummationByPartsOperators"],
           mode = PKGMODE_MANIFEST)
