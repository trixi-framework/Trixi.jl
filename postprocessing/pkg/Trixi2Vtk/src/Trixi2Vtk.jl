module Trixi2Vtk

# Number of spatial dimensions
const ndim = 2
export ndim

# Include all top-level submodule files
include("auxiliary.jl")
include("interpolate.jl")
include("io.jl")
include("pointlocators.jl")
include("vtktools.jl")

# Include top-level run method
include("run.jl")


end # module Trixi2Vtk
