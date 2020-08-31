"""
    Trixi

**Trixi.jl** is a flexible numerical simulation framework for partial
differential equations. It is based on a two-dimensional hierarchical mesh
(quadtree) and supports several governing equations such as the compressible Euler
equations, magnetohydrodynamics equations, or hyperbolic diffusion equations.
Trixi is written in Julia and aims to be easy to use and
extend also for new or inexperienced users.

See also: [trixi-framework/Trixi.jl](https://github.com/trixi-framework/Trixi.jl)
"""
module Trixi

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
export CompressibleEulerEquations2D,
       IdealGlmMhdEquations2D,
       HyperbolicDiffusionEquations2D,
       LinearScalarAdvectionEquation2D, LinearScalarAdvectionEquation3D
export flux_central, flux_lax_friedrichs,
       flux_chandrashekar, flux_ranocha, flux_derigs_etal, flux_kennedy_gruber, flux_shima_etal
export examples_dir, get_examples, default_example


end
