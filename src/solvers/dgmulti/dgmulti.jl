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

# parabolic terms for DGMulti solvers
include("dg_parabolic.jl")
