module Jul1dge

using Reexport
const ndim = 1
export ndim

include("SysEqn.jl")
include("mesh/mesh.jl")
include("dg/dg.jl")
include("io/io.jl")
include("timedisc/timedisc.jl")

@reexport using .SysEqnMod
@reexport using .MeshMod
@reexport using .DgMod
@reexport using .IoMod

end
