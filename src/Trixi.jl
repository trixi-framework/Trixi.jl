"""
    Trixi

**Trixi.jl** is a numerical simulation framework for hyperbolic conservation
laws. A key objective for the
framework is to be useful to both scientists and students. Therefore, next to
having an extensible design with a fast implementation, Trixi is
focused on being easy to use for new or inexperienced users, including the
installation and postprocessing procedures.

See also: [trixi-framework/Trixi.jl](https://github.com/trixi-framework/Trixi.jl)
"""
module Trixi

# Include other packages that are used in Trixi
# (standard library packages first, other packages next, all of them sorted alphabetically)
using LinearAlgebra: dot
using Pkg.TOML: parsefile
using Printf: @printf, @sprintf, println
using Profile: clear_malloc_data
using Random: seed!

using DiffEqBase: ODEProblem, ODESolution, get_du, u_modified!, set_proposed_dt!
using DiffEqCallbacks: CallbackSet, DiscreteCallback
using EllipsisNotation # ..
using HDF5: h5open, attrs
using StaticArrays: @MVector, @SVector, MVector, MMatrix, MArray, SVector, SMatrix, SArray
using TimerOutputs: @notimeit, @timeit, @timeit_debug, TimerOutput, print_timer, reset_timer!
using UnPack: @unpack

# Tullio.jl makes use of LoopVectorization.jl via Requires.jl.
# Hence, we need `using LoopVectorization` after loading Tullio and before using `@tullio`.
using Tullio: @tullio
using LoopVectorization


# Use a central dictionary for global settings
const globals = Dict{Symbol, Any}()
export globals

# Basic abstract types creating the hierarchy
abstract type AbstractEquations{NDIMS, NVARS} end



# Include all top-level source files
include("auxiliary/auxiliary.jl")
include("equations/equations.jl")
include("mesh/mesh.jl")
include("solvers/solvers.jl")
include("semidiscretization.jl")
include("semidiscretization_euler_gravity.jl")
include("io/io.jl")
include("timedisc/timedisc.jl")
include("amr/amr.jl")
include("callbacks/callbacks.jl")

# Include top-level run method
include("run.jl")


# export types/functions that define the public API of Trixi
export CompressibleEulerEquations2D, CompressibleEulerEquations3D,
       IdealGlmMhdEquations2D, IdealGlmMhdEquations3D,
       HyperbolicDiffusionEquations2D, HyperbolicDiffusionEquations3D,
       LinearScalarAdvectionEquation2D, LinearScalarAdvectionEquation3D

export flux_central, flux_lax_friedrichs, flux_hll,
       flux_chandrashekar, flux_ranocha, flux_derigs_etal, flux_kennedy_gruber, flux_shima_etal

# TODO: Taal decide, which initial conditions and source terms will be used/exported
export initial_conditions_convergence_test,
       initial_conditions_weak_blast_wave, initial_conditions_gauss

export source_terms_harmonic

export TreeMesh

export DGSEM,
       VolumeIntegralWeakForm, VolumeIntegralFluxDifferencing

export SemidiscretizationHyperbolic, semidiscretize, compute_coefficients

export SemidiscretizationEulerGravity, ParametersEulerGravity, timestep_gravity_erk52_3Sstar!

export AliveCallback, AnalysisCallback, SaveSolutionCallback, StepsizeCallback

export entropy, energy_total

export examples_dir, get_examples, default_example


end
