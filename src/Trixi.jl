module Trixi

# Use README as the docstring of the module:
@doc read(joinpath(dirname(@__DIR__), "README.md"), String) Trixi

# Include other packages that are used in Trixi
# (standard library packages first, other packages next, all of them sorted alphabetically)
using Pkg.TOML: parsefile
using Printf: @printf, @sprintf, println
using Profile: clear_malloc_data
using Random: seed!

using HDF5: h5open, attrs
using StaticArrays: @MVector, @SVector, MVector, MMatrix, MArray, SVector, SMatrix, SArray
using TimerOutputs: @notimeit, @timeit, TimerOutput, print_timer, reset_timer!
using UnPack: @unpack


# Set ndim as a short, module-wide constant.
# Rationale: This makes code easier to understand than using hardcoded dimension values.
"""
    ndim

Specify the number of spatial dimensions.

Always use `ndim` instead of hard-coding the literal `2` when referring to the
number of dimensions. This makes code easier to understand, since it adds a
meaning to the number.
"""
const ndim = 2
export ndim

# Use a central dictionary for global settings
const globals = Dict{Symbol, Any}()
export globals

# Include all top-level source files
include("auxiliary/auxiliary.jl")
include("equations/equations.jl")
include("mesh/mesh.jl")
include("solvers/solvers.jl")
include("io/io.jl")
include("timedisc/timedisc.jl")
include("amr/amr.jl")

# Include top-level run method
include("run.jl")


# export types/functions that define the public API of Trixi
export CompressibleEulerEquations, IdealGlmMhdEquations, HyperbolicDiffusionEquations, LinearScalarAdvectionEquation
export flux_central, flux_lax_friedrichs,
       flux_chandrashekar, flux_ranocha, flux_derigs_etal, flux_kennedy_gruber, flux_kuya_etal


end
