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
using Random: seed!

using HDF5: h5open, attrs
using MPI # We use all symbols, but for now we always prefix with `MPI.`, e.g., `MPI.Init()`
using OffsetArrays: OffsetArray, OffsetVector
using StaticArrays: @MVector, @SVector, MVector, MMatrix, MArray, SVector, SMatrix, SArray
using TimerOutputs: @notimeit, @timeit, TimerOutput, print_timer, reset_timer!
using UnPack: @unpack

# Tullio.jl makes use of LoopVectorization.jl via Requires.jl.
# Hence, we need `using LoopVectorization` after loading Tullio and before using @tullio.
using Tullio: @tullio
using LoopVectorization

# Use a central dictionary for global settings
const globals = Dict{Symbol, Any}()
export globals

# Include all top-level source files
include("auxiliary/auxiliary.jl")
include("parallel/parallel.jl")
include("equations/equations.jl")
include("mesh/mesh.jl")
include("solvers/solvers.jl")
include("io/io.jl")
include("timedisc/timedisc.jl")
include("amr/amr.jl")

# Include top-level run method
include("run.jl")


# export types/functions that define the public API of Trixi
export CompressibleEulerEquations2D, CompressibleEulerEquations3D,
       IdealGlmMhdEquations2D, IdealGlmMhdEquations3D,
       HyperbolicDiffusionEquations2D, HyperbolicDiffusionEquations3D,
       LinearScalarAdvectionEquation2D, LinearScalarAdvectionEquation3D
export flux_central, flux_lax_friedrichs, flux_hll,
       flux_chandrashekar, flux_ranocha, flux_derigs_etal, flux_kennedy_gruber, flux_shima_etal
export examples_dir, get_examples, default_example


function __init__()
  # Initialize MPI
  init_mpi()

  # Initialize global MPI state
  MPI_RANK[] = MPI.Comm_rank(MPI.COMM_WORLD)
  MPI_SIZE[] = MPI.Comm_size(MPI.COMM_WORLD)
  MPI_IS_PARALLEL[] = MPI_SIZE[] > 1
  MPI_IS_SERIAL[] = !MPI_IS_PARALLEL[]
  MPI_IS_ROOT[] = MPI_IS_SERIAL[] || MPI_RANK[] == 0

  # Initialize methods for dispatching on parallel execution
  if MPI_IS_PARALLEL[]
    eval(:(mpi_parallel() = Val(true)))
  else
    eval(:(mpi_parallel() = Val(false)))
  end
end


end
