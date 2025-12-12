#src # Other SBP schemes (FD, CGSEM) via `DGMulti` solver

# For a tutorial about DG schemes via the [`DGMulti`](@ref) solver please visit the [previous tutorial](@ref DGMulti_1).
# The `DGMulti` solver also supports other methods than DG. The important property a method has to
# fulfill is the summation-by-parts (SBP) property. The package [SummationByPartsOperators.jl](https://github.com/ranocha/SummationByPartsOperators.jl)
# provides such methods, like a finite difference SBP (FD SBP) scheme. To do this,
# you need to create an SBP derivative operator and pass that as `approximation_type`
# to the `DGMulti` constructor. For example, the classical second-order FD SBP operator
# can be created as
using Trixi.SummationByPartsOperators # or add SummationByPartsOperators to your project and use it directly
D = derivative_operator(MattssonNordström2004(), derivative_order = 1, accuracy_order = 2,
                        xmin = 0.0, xmax = 1.0, N = 11)
# Here, the arguments `xmin` and `xmax` do not matter beyond setting the real type
# used for the operator - they just set a reference element and are rescaled on the
# physical elements. The parameter `N` determines the number of finite difference nodes.
# Then, `D` can be used as `approximation_type` like `SBP()` in a multi-block fashion.
# In multiple dimensions, such a 1D SBP operator will be used in a tensor product fashion,
# i.e., in each coordinate direction. In particular, you can use them only on 1D, 2D `Quad()`,
# and 3D `Hex()` elements.
#
# You can also use fully periodic single-block FD methods by creating a periodic SBP
# operator. For example, a fully periodic FD operator can be constructed as
D = periodic_derivative_operator(derivative_order = 1, accuracy_order = 2,
                                 xmin = 0.0, xmax = 1.0, N = 11)
# An example using such an FD method is implemented in
# [`elixir_euler_fdsbp_periodic.jl`](https://github.com/trixi-framework/Trixi.jl/blob/main/examples/dgmulti_2d/elixir_euler_fdsbp_periodic.jl).
# For all parameters and other calling options, please have a look in the
# [documentation of SummationByPartsOperators.jl](https://ranocha.de/SummationByPartsOperators.jl/stable/).

# Another possible method is for instance a continuous Galerkin (CGSEM) method. You can use such a
# method with polynomial degree of `3` (`N=4` Legendre Lobatto nodes on `[0, 1]`) coupled continuously
# on a uniform mesh with `Nx=10` elements by setting `approximation_type` to
using Trixi.SummationByPartsOperators # or add SummationByPartsOperators to your project and use it directly
D = couple_continuously(legendre_derivative_operator(xmin = 0.0, xmax = 1.0, N = 4),
                        UniformPeriodicMesh1D(xmin = -1.0, xmax = 1.0, Nx = 10))

# To choose a discontinuous coupling (DGSEM), use `couple_discontinuously()` instead of `couple_continuously()`.

# For more information and other SBP operators, see the documentations of [StartUpDG.jl](https://jlchan.github.io/StartUpDG.jl/dev/)
# and [SummationByPartsOperators.jl](https://ranocha.de/SummationByPartsOperators.jl/stable/).

# ## Package versions

# These results were obtained using the following versions.

using InteractiveUtils
versioninfo()

using Pkg
Pkg.status(["Trixi", "StartUpDG", "SummationByPartsOperators"],
           mode = PKGMODE_MANIFEST)
