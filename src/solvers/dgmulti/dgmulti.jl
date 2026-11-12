# basic types and functions for DGMulti solvers
include("types.jl")
include("dg.jl")

# flux differencing solver routines for DGMulti solvers
include("flux_differencing_gauss_sbp.jl")
include("flux_differencing.jl")

# adaptive volume integral solver
include("volume_integral_adaptive.jl")

# integration of SummationByPartsOperators.jl
include("sbp.jl")

# specialization of DGMulti to specific equations
include("flux_differencing_compressible_euler.jl")

# shock capturing
include("shock_capturing.jl")

# algebraic low order method on SBP nodes, used by shock capturing and entropy correction
include("volume_integral_low_order.jl")

# parabolic terms for DGMulti solvers
include("dg_parabolic.jl")

# cell averages, used by the positivity-preserving stage callbacks
include("utils_u_mean.jl")
