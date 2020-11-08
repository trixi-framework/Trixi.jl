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
using Pkg.TOML: parsefile, parse
using Printf: @printf, @sprintf, println
using Profile: clear_malloc_data
using Random: seed! # TODO: Taal, can be removed

import DiffEqBase: ODEProblem, ODESolution, get_du, u_modified!, set_proposed_dt!, terminate!
using DiffEqCallbacks: CallbackSet, DiscreteCallback
using EllipsisNotation # ..
using HDF5: h5open, attrs
using LinearMaps: LinearMap
import MPI
using OffsetArrays: OffsetArray, OffsetVector
using StaticArrays: @MVector, @SVector, MVector, MMatrix, MArray, SVector, SMatrix, SArray
using TimerOutputs: @notimeit, @timeit, @timeit_debug, TimerOutput, print_timer, reset_timer!
using UnPack: @unpack

# Tullio.jl makes use of LoopVectorization.jl via Requires.jl.
# Hence, we need `using LoopVectorization` after loading Tullio and before using `@tullio`.
using Tullio: @tullio
using LoopVectorization


# TODO: Taal remove globals
# Use a central dictionary for global settings
const globals = Dict{Symbol, Any}()
export globals


# Define the entry points of our type hierarchy, e.g.
#     AbstractEquations, AbstractSemidiscretization etc.
# Placing them here allows us to make use of them for dispatch even for
# other stuff defined very early in our include pipeline, e.g.
#     IndicatorLöhner(semi::AbstractSemidiscretization)
include("basic_types.jl")

# Include all top-level source files
include("auxiliary/auxiliary.jl")
include("parallel/parallel.jl")
include("equations/equations.jl")
include("mesh/mesh.jl")
include("solvers/solvers.jl")
include("semidiscretization.jl")
include("io/io.jl")
include("timedisc/timedisc.jl")
include("amr/amr.jl")
include("callbacks/callbacks.jl")
include("semidiscretization_euler_gravity.jl")

# TODO: Taal refactor, get rid of old run methods, rename the file
# Include top-level run method
include("run.jl")


# export types/functions that define the public API of Trixi
export CompressibleEulerEquations1D, CompressibleEulerEquations2D, CompressibleEulerEquations3D,
       IdealGlmMhdEquations2D, IdealGlmMhdEquations3D,
       HyperbolicDiffusionEquations2D, HyperbolicDiffusionEquations3D,
       LinearScalarAdvectionEquation1D, LinearScalarAdvectionEquation2D, LinearScalarAdvectionEquation3D

export flux_central, flux_lax_friedrichs, flux_hll, flux_upwind,
       flux_chandrashekar, flux_ranocha, flux_derigs_etal, flux_kennedy_gruber, flux_shima_etal

export initial_condition_constant,
       initial_condition_gauss,
       initial_condition_density_wave,
       initial_condition_isentropic_vortex,
       initial_condition_khi,
       initial_condition_weak_blast_wave, initial_condition_blast_wave,
       initial_condition_sedov_blast_wave, initial_condition_medium_sedov_blast_wave,
       initial_condition_blob,
       initial_condition_orszag_tang,
       initial_condition_rotor

export boundary_condition_periodic,
       boundary_condition_gauss

export initial_condition_convergence_test, source_terms_convergence_test, boundary_condition_convergence_test
export initial_condition_harmonic_nonperiodic, source_terms_harmonic, boundary_condition_harmonic_nonperiodic
export initial_condition_poisson_periodic, source_terms_poisson_periodic
export initial_condition_poisson_nonperiodic, source_terms_poisson_nonperiodic, boundary_condition_poisson_nonperiodic
export initial_condition_sedov_self_gravity, boundary_condition_sedov_self_gravity
export initial_condition_eoc_test_coupled_euler_gravity, source_terms_eoc_test_coupled_euler_gravity, source_terms_eoc_test_euler

export TreeMesh

export DG,
       DGSEM, LobattoLegendreBasis,
       VolumeIntegralWeakForm, VolumeIntegralFluxDifferencing,
       VolumeIntegralShockCapturingHG, IndicatorHennemannGassner,
       MortarL2

export nelements, nnodes, nvariables,
       eachelement, eachnode, eachvariable

export SemidiscretizationHyperbolic, semidiscretize, compute_coefficients, integrate

export SemidiscretizationEulerGravity, ParametersEulerGravity,
       timestep_gravity_erk52_3Sstar!, timestep_gravity_carpenter_kennedy_erk54_2N!

export SummaryCallback, SteadyStateCallback, AMRCallback, StepsizeCallback,
       SaveRestartCallback, SaveSolutionCallback, AnalysisCallback, AliveCallback,
       TrivialCallback

export load_mesh, load_time

export ControllerThreeLevel, ControllerThreeLevelCombined,
       IndicatorLöhner, IndicatorLoehner, IndicatorMax
export density, pressure, density_pressure

export entropy, energy_total, energy_kinetic, energy_internal, energy_magnetic, cross_helicity

export trixi_include, examples_dir, get_examples, default_example

export convergence_test, jacobian_fd, linear_structure


function __init__()
  init_mpi()
end


end
