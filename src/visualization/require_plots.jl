# This file contains all relevant code (or includes it from other files) that only works if the
# Plots.jl package is available. This file itself is included using Requires.jl on the condition
# that the Plots package is loaded.
using .Plots: plot, plot!

include("callback.jl")
