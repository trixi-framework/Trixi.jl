module Jul1dge

using Reexport
const ndim = 1
export ndim

include("equation/equation.jl")
include("mesh/mesh.jl")
include("dg/dg.jl")
# include("io/io.jl")
include("timedisc/timedisc.jl")

@reexport using .Equation
@reexport using .MeshMod
@reexport using .DgMod
# @reexport using .IoMod
@reexport using .TimeDisc

end
