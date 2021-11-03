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

using LinearAlgebra: LinearAlgebra, diag, diagm, dot, mul!, norm, cross, normalize, I, UniformScaling
using Printf: @printf, @sprintf, println

# import @reexport now to make it available for further imports/exports
using Reexport: @reexport

using DiffEqBase: @muladd, CallbackSet, DiscreteCallback,
                  ODEProblem, ODESolution, ODEFunction
import DiffEqBase: get_du, get_tmp_cache, u_modified!,
                   AbstractODEIntegrator, init, step!, check_error,
                   get_proposed_dt, set_proposed_dt!,
                   terminate!, remake
using CodeTracking: code_string
@reexport using EllipsisNotation # ..
using ForwardDiff: ForwardDiff
using HDF5: h5open, attributes
using LinearMaps: LinearMap
using LoopVectorization: LoopVectorization, @turbo, indices
using LoopVectorization.ArrayInterface: static_length
using MPI: MPI
using GeometryBasics: GeometryBasics
using Octavian: Octavian, matmul!
using Polyester: @batch # You know, the cheapest threads you can find...
using OffsetArrays: OffsetArray, OffsetVector
using P4est
using Setfield: @set
using RecipesBase: RecipesBase
using Requires: @require
using SparseArrays: AbstractSparseMatrix, sparse, droptol!, rowvals, nzrange
using Static: One
@reexport using StaticArrays: SVector
using StaticArrays: MVector, MArray, SMatrix
using StrideArrays: PtrArray, StrideArray, StaticInt
@reexport using StructArrays: StructArrays, StructArray
using TimerOutputs: TimerOutputs, @notimeit, TimerOutput, print_timer, reset_timer!
using Triangulate: Triangulate, TriangulateIO, triangulate
using TriplotBase: TriplotBase
using TriplotRecipes: DGTriPseudocolor
@reexport using UnPack: @unpack
using UnPack: @pack!

# finite difference SBP operators
using SummationByPartsOperators: AbstractDerivativeOperator, DerivativeOperator, grid
import SummationByPartsOperators: integrate, left_boundary_weight, right_boundary_weight
@reexport using SummationByPartsOperators:
  SummationByPartsOperators, derivative_operator

# DGMulti solvers
@reexport using StartUpDG: StartUpDG, Polynomial, SBP, Line, Tri, Quad, Hex, Tet
using StartUpDG: RefElemData, MeshData, AbstractElemShape

# TODO: include_optimized
# This should be used everywhere (except to `include("interpolations.jl")`)
# once the upstream issue https://github.com/timholy/Revise.jl/issues/634
# is fixed; tracked in https://github.com/trixi-framework/Trixi.jl/issues/664.
# # By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# # Since these FMAs can increase the performance of many numerical algorithms,
# # we need to opt-in explicitly.
# # See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
# function include_optimized(filename)
#   include(expr -> quote @muladd begin $expr end end, filename)
# end


# Define the entry points of our type hierarchy, e.g.
#     AbstractEquations, AbstractSemidiscretization etc.
# Placing them here allows us to make use of them for dispatch even for
# other stuff defined very early in our include pipeline, e.g.
#     IndicatorLöhner(semi::AbstractSemidiscretization)
include("basic_types.jl")

# Include all top-level source files
include("auxiliary/auxiliary.jl")
include("auxiliary/mpi.jl")
include("auxiliary/p4est.jl")
include("equations/equations.jl")
include("meshes/meshes.jl")
include("solvers/solvers.jl")
include("semidiscretization/semidiscretization.jl")
include("semidiscretization/semidiscretization_hyperbolic.jl")
include("semidiscretization/semidiscretization_euler_acoustics.jl")
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
       LatticeBoltzmannEquations2D, LatticeBoltzmannEquations3D,
       ShallowWaterEquations2D

export flux, flux_central, flux_lax_friedrichs, flux_hll, flux_hllc, flux_godunov,
       flux_chandrashekar, flux_ranocha, flux_derigs_etal, flux_hindenlang_gassner,
       flux_nonconservative_powell,
       flux_kennedy_gruber, flux_shima_etal, flux_ec,
       flux_fjordholm_etal, flux_nonconservative_fjordholm_etal,
       flux_wintermeyer_etal, flux_nonconservative_wintermeyer_etal,
       FluxPlusDissipation, DissipationGlobalLaxFriedrichs, DissipationLocalLaxFriedrichs,
       FluxLaxFriedrichs, max_abs_speed_naive,
       FluxHLL, min_max_speed_naive,
       FluxRotated

export initial_condition_constant,
       initial_condition_gauss,
       initial_condition_density_wave,
       initial_condition_weak_blast_wave

export boundary_condition_periodic,
       BoundaryConditionDirichlet,
       boundary_condition_noslip_wall,
       boundary_condition_slip_wall,
       boundary_condition_wall

export initial_condition_convergence_test, source_terms_convergence_test
export source_terms_harmonic
export initial_condition_poisson_nonperiodic, source_terms_poisson_nonperiodic, boundary_condition_poisson_nonperiodic
export initial_condition_eoc_test_coupled_euler_gravity, source_terms_eoc_test_coupled_euler_gravity, source_terms_eoc_test_euler

export cons2cons, cons2prim, prim2cons, cons2macroscopic, cons2state, cons2mean,
       cons2entropy, entropy2cons
export density, pressure, density_pressure, velocity, global_mean_vars, equilibrium_distribution, waterheight_pressure
export entropy, energy_total, energy_kinetic, energy_internal, energy_magnetic, cross_helicity
export lake_at_rest_error
export ncomponents, eachcomponent

export TreeMesh, StructuredMesh, UnstructuredMesh2D, P4estMesh

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

export SemidiscretizationEulerAcoustics

export SemidiscretizationEulerGravity, ParametersEulerGravity,
       timestep_gravity_erk52_3Sstar!, timestep_gravity_carpenter_kennedy_erk54_2N!

export SummaryCallback, SteadyStateCallback, AnalysisCallback, AliveCallback,
       SaveRestartCallback, SaveSolutionCallback, TimeSeriesCallback, VisualizationCallback,
       AveragingCallback,
       AMRCallback, StepsizeCallback,
       GlmSpeedCallback, LBMCollisionCallback, EulerAcousticsCouplingCallback,
       TrivialCallback

export load_mesh, load_time

export ControllerThreeLevel, ControllerThreeLevelCombined,
       IndicatorLöhner, IndicatorLoehner, IndicatorMax,
       IndicatorNeuralNetwork, NeuralNetworkPerssonPeraire, NeuralNetworkRayHesthaven, NeuralNetworkCNN

export PositivityPreservingLimiterZhangShu

export trixi_include, examples_dir, get_examples, default_example, default_example_unstructured

export convergence_test, jacobian_fd, jacobian_ad_forward, linear_structure

export DGMulti, AbstractMeshData, VertexMappedMesh, estimate_dt
export GaussSBP

# Visualization-related exports
export PlotData1D, PlotData2D, ScalarPlotData2D, getmesh, adapt_to_mesh_level!, adapt_to_mesh_level

function __init__()
  init_mpi()

  init_p4est()

  # Enable features that depend on the availability of the Plots package
  @require Plots="91a5bcdd-55d7-5caf-9e0b-520d859cae80" begin
    using .Plots: Plots
  end

  @require Makie="ee78f7c6-11fb-53f2-987a-cfe4a2b5a57a" begin
    include("visualization/recipes_makie.jl")
    using .Makie: Makie
    export iplot, iplot! # interactive plot
  end

  @require Flux="587475ba-b771-5e3f-ad9e-33799f191a9c" begin
    using Flux: params
  end

  # FIXME upstream. This is a hacky workaround for
  #       https://github.com/trixi-framework/Trixi.jl/issues/628
  # The related upstream issues appear to be
  #       https://github.com/JuliaLang/julia/issues/35800
  #       https://github.com/JuliaLang/julia/issues/32552
  #       https://github.com/JuliaLang/julia/issues/41740
  # See also https://discourse.julialang.org/t/performance-depends-dramatically-on-compilation-order/58425
  let
    for T in (Float32, Float64)
      u_mortars_2d = zeros(T, 2, 2, 2, 2, 2)
      view(u_mortars_2d, 1, :, 1, :, 1)
      u_mortars_3d = zeros(T, 2, 2, 2, 2, 2, 2)
      view(u_mortars_3d, 1, :, 1, :, :, 1)
    end
  end
end


include("auxiliary/precompile.jl")
_precompile_manual_()


end
