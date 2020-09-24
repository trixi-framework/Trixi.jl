
# include callback definitions in their preferred order
# when called after a complete step
include("amr.jl")
include("summary.jl")
include("stepsize.jl")
include("analysis.jl")
include("save_solution.jl")
include("alive.jl")
