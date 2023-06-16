# includes solver files for DGMulti solvers
include("dgmulti/types.jl")
include("dgmulti/dg.jl")
include("dgmulti/flux_differencing_gauss_sbp.jl")
include("dgmulti/flux_differencing.jl")

# integration of SummationByPartsOperators.jl
include("dgmulti/sbp.jl")

# specialization of DGMulti to specific equations
include("dgmulti/flux_differencing_compressible_euler.jl")

# shock capturing
include("dgmulti/shock_capturing.jl")

# parabolic terms for DGMulti solvers
include("dgmulti/dg_parabolic.jl")
