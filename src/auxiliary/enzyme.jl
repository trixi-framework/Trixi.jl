# Support for automatic differentiation with Enzyme.jl.

# The time step size of the CFL-based `StepsizeCallback` is not differentiated. Hence,
# reverse-mode derivatives through a time integration use the step sizes of the primal
# run as constants ("frozen" step size control), as in discrete adjoints of CFD codes.
EnzymeCore.EnzymeRules.inactive(::typeof(max_dt), args...) = nothing
