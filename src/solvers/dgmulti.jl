# includes solver files for DGMulti solvers
include("dgmulti/types.jl")
include("dgmulti/dg.jl")
include("dgmulti/flux_differencing_gauss_sbp.jl")
include("dgmulti/flux_differencing.jl")

# specialization of DGMulti to specific equations
include("dgmulti/flux_differencing_compressible_euler.jl")
