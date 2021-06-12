"""
    Trixi

**Trixi.jl** is a numerical simulation framework for hyperbolic conservation
laws. A key objective for the framework is to be useful to both scientists
and students. Therefore, next to having an extensible design with a fast
implementation, Trixi is focused on being easy to use for new or inexperienced
users, including the installation and postprocessing procedures.

To get started, run your first simulation with Trixi using

    trixi_include(default_example())

See also: [trixi-framework/Trixi.jl](https://github.com/trixi-framework/Trixi.jl)
"""
module Trixi

# Include other packages that are used in Trixi
# (standard library packages first, other packages next, all of them sorted alphabetically)

using LinearAlgebra: diag, dot, mul!, norm, cross, normalize
using Printf: @printf, @sprintf, println

# import @reexport now to make it available for further imports/exports
using Reexport: @reexport

import DiffEqBase: CallbackSet, DiscreteCallback,
                   ODEProblem, ODESolution, ODEFunction,
                   get_du, get_tmp_cache, u_modified!,
                   get_proposed_dt, set_proposed_dt!, terminate!, remake
using CodeTracking: code_string
@reexport using EllipsisNotation # ..
import ForwardDiff
using HDF5: h5open, attributes
using LinearMaps: LinearMap
using LoopVectorization: LoopVectorization, @turbo, indices
using LoopVectorization.ArrayInterface: static_length
import MPI
using Polyester: @batch # You know, the cheapest threads you can find...
using OffsetArrays: OffsetArray, OffsetVector
using P4est
using RecipesBase
using Requires
@reexport using StaticArrays: SVector
using StaticArrays: MVector, MArray, SMatrix
using StrideArrays: PtrArray, StrideArray, StaticInt
using TimerOutputs: TimerOutputs, @notimeit, TimerOutput, print_timer, reset_timer!
@reexport using UnPack: @unpack
using UnPack: @pack!

# finite difference SBP operators
using SummationByPartsOperators: AbstractDerivativeOperator, DerivativeOperator, grid
import SummationByPartsOperators: integrate, left_boundary_weight, right_boundary_weight
@reexport using SummationByPartsOperators:
  SummationByPartsOperators, derivative_operator

# Define the entry points of our type hierarchy, e.g.
#     AbstractEquations, AbstractSemidiscretization etc.
# Placing them here allows us to make use of them for dispatch even for
# other stuff defined very early in our include pipeline, e.g.
#     IndicatorLöhner(semi::AbstractSemidiscretization)
include("basic_types.jl")

# Include all top-level source files
include("auxiliary/auxiliary.jl")
include("auxiliary/mpi.jl")
include("equations/equations.jl")
include("mesh/mesh.jl")
include("solvers/solvers.jl")
include("semidiscretization/semidiscretization.jl")
include("semidiscretization/semidiscretization_hyperbolic.jl")
include("callbacks_step/callbacks_step.jl")
include("callbacks_stage/callbacks_stage.jl")
include("semidiscretization/semidiscretization_euler_gravity.jl")
include("time_integration/time_integration.jl")

# `trixi_include` and special elixirs such as `convergence_test`
include("auxiliary/special_elixirs.jl")

# Plot recipes and conversion functions to visualize results with Plots.jl
include("visualization/visualization.jl")


# export types/functions that define the public API of Trixi

export AcousticPerturbationEquations2D,
       CompressibleEulerEquations1D, CompressibleEulerEquations2D, CompressibleEulerEquations3D,
       CompressibleEulerMulticomponentEquations1D, CompressibleEulerMulticomponentEquations2D,
       IdealGlmMhdEquations1D, IdealGlmMhdEquations2D, IdealGlmMhdEquations3D,
       IdealGlmMhdMulticomponentEquations1D, IdealGlmMhdMulticomponentEquations2D,
       HyperbolicDiffusionEquations1D, HyperbolicDiffusionEquations2D, HyperbolicDiffusionEquations3D,
       LinearScalarAdvectionEquation1D, LinearScalarAdvectionEquation2D, LinearScalarAdvectionEquation3D,
       InviscidBurgersEquation1D,
       LatticeBoltzmannEquations2D, LatticeBoltzmannEquations3D

export flux, flux_central, flux_lax_friedrichs, flux_hll, flux_hllc, flux_godunov,
       flux_chandrashekar, flux_ranocha, flux_derigs_etal, flux_hindenlang,
       flux_kennedy_gruber, flux_shima_etal, flux_ec,
       FluxPlusDissipation, DissipationGlobalLaxFriedrichs, DissipationLocalLaxFriedrichs,
       FluxLaxFriedrichs, max_abs_speed_naive,
       FluxHLL, min_max_speed_naive,
       FluxRotated

export initial_condition_constant,
       initial_condition_gauss,
       initial_condition_density_wave, initial_condition_density_pulse,
       initial_condition_isentropic_vortex,
       initial_condition_khi,
       initial_condition_weak_blast_wave, initial_condition_blast_wave,
       initial_condition_sedov_blast_wave, initial_condition_medium_sedov_blast_wave,
       initial_condition_two_interacting_blast_waves, boundary_condition_two_interacting_blast_waves,
       initial_condition_blob,
       initial_condition_orszag_tang,
       initial_condition_rotor,
       initial_condition_shock_bubble,
       initial_condition_taylor_green_vortex

export boundary_condition_periodic,
       BoundaryConditionDirichlet,
       boundary_condition_wall_noslip,
       boundary_condition_wall,
       boundary_condition_zero,
       BoundaryConditionWall,
       boundary_state_slip_wall

export initial_condition_convergence_test, source_terms_convergence_test
export initial_condition_harmonic_nonperiodic, source_terms_harmonic
export initial_condition_poisson_periodic, source_terms_poisson_periodic
export initial_condition_poisson_nonperiodic, source_terms_poisson_nonperiodic, boundary_condition_poisson_nonperiodic
export initial_condition_briowu_shock_tube, initial_condition_torrilhon_shock_tube, initial_condition_ryujones_shock_tube,
       initial_condition_shu_osher_shock_tube, initial_condition_shu_osher_shock_tube_flipped
export initial_condition_sedov_self_gravity, boundary_condition_sedov_self_gravity
export initial_condition_eoc_test_coupled_euler_gravity, source_terms_eoc_test_coupled_euler_gravity, source_terms_eoc_test_euler
export initial_condition_lid_driven_cavity, boundary_condition_lid_driven_cavity
export initial_condition_couette_steady, initial_condition_couette_unsteady, boundary_condition_couette
export initial_condition_gauss_wall
export initial_condition_monopole, boundary_condition_monopole

export cons2cons, cons2prim, prim2cons, cons2macroscopic, cons2state, cons2mean,
       cons2entropy, entropy2cons
export density, pressure, density_pressure, velocity
export entropy, energy_total, energy_kinetic, energy_internal, energy_magnetic, cross_helicity

export TreeMesh, CurvedMesh, UnstructuredQuadMesh, P4estMesh

export DG,
       DGSEM, LobattoLegendreBasis,
       VolumeIntegralWeakForm, VolumeIntegralStrongForm,
       VolumeIntegralFluxDifferencing,
       VolumeIntegralPureLGLFiniteVolume,
       VolumeIntegralShockCapturingHG, IndicatorHennemannGassner,
       SurfaceIntegralWeakForm, SurfaceIntegralStrongForm,
       MortarL2

export nelements, nnodes, nvariables,
       eachelement, eachnode, eachvariable

export SemidiscretizationHyperbolic, semidiscretize, compute_coefficients, integrate

export SemidiscretizationEulerGravity, ParametersEulerGravity,
       timestep_gravity_erk52_3Sstar!, timestep_gravity_carpenter_kennedy_erk54_2N!

export SummaryCallback, SteadyStateCallback, AnalysisCallback, AliveCallback,
       SaveRestartCallback, SaveSolutionCallback, TimeSeriesCallback, VisualizationCallback,
       AMRCallback, StepsizeCallback,
       GlmSpeedCallback, LBMCollisionCallback,
       TrivialCallback

export load_mesh, load_time

export ControllerThreeLevel, ControllerThreeLevelCombined,
       IndicatorLöhner, IndicatorLoehner, IndicatorMax

export PositivityPreservingLimiterZhangShu

export trixi_include, examples_dir, get_examples, default_example, default_example_unstructured

export convergence_test, jacobian_fd, jacobian_ad_forward, linear_structure

# Visualization-related exports
export PlotData1D, PlotData2D, getmesh, adapt_to_mesh_level!, adapt_to_mesh_level


function __init__()
  init_mpi()

  init_p4est()

  # Enable features that depend on the availability of the Plots package
  @require Plots="91a5bcdd-55d7-5caf-9e0b-520d859cae80" begin
    using .Plots: plot, plot!, savefig
  end
end


include("auxiliary/precompile.jl")
_precompile_manual_()


end
