module Jul1dge

const ndim = 1
export ndim

include("SysEqn.jl")
include("Mesh.jl")
include("Common.jl")
include("Dg.jl")
include("Io.jl")

using Reexport
@reexport using .SysEqnMod
@reexport using .MeshMod
@reexport using .CommonMod
@reexport using .DgMod
@reexport using .IoMod

end
