module Trixi

# Set ndim as a short, module-wide constant.
# Rationale: This makes code easier to understand than using hardcoded dimension values.
const ndim = 2
export ndim

# Include all top-level submodule files
include("auxiliary/auxiliary.jl")
include("equations/equations.jl")
include("mesh/mesh.jl")
include("solvers/solvers.jl")
include("io/io.jl")
include("timedisc/timedisc.jl")

# Include top-level run method
include("run.jl")

end
