
"""
    init_mpi()

Initialize MPI by calling `MPI.Initialized()`. The function will check if MPI is already initialized
and if yes, do nothing, thus it is safe to call it multiple times.
"""
function init_mpi()
  if MPI_INITIALIZED[]
    return nothing
  end

  # MPI.jl handles multiple calls to MPI.Init appropriately. Thus, we don't need
  # any common checks of the form `if MPI.Initialized() ...`.
  # threadlevel=MPI.THREAD_FUNNELED: Only main thread makes MPI calls
  # finalize_atexit=true           : MPI.jl will call call MPI.Finalize as `atexit` hook
  provided = MPI.Init(threadlevel=MPI.THREAD_FUNNELED, finalize_atexit=true)
  @assert provided >= MPI.THREAD_FUNNELED "MPI library with insufficient threading support"

  # Initialize global MPI state
  MPI_RANK[] = MPI.Comm_rank(MPI.COMM_WORLD)
  MPI_SIZE[] = MPI.Comm_size(MPI.COMM_WORLD)
  MPI_IS_PARALLEL[] = MPI_SIZE[] > 1
  MPI_IS_SERIAL[] = !MPI_IS_PARALLEL[]
  MPI_IS_ROOT[] = MPI_IS_SERIAL[] || MPI_RANK[] == 0
  MPI_INITIALIZED[] = true

  return nothing
end


const MPI_INITIALIZED = Ref(false)
const MPI_RANK = Ref(-1)
const MPI_SIZE = Ref(-1)
const MPI_IS_PARALLEL = Ref(false)
const MPI_IS_SERIAL = Ref(true)
const MPI_IS_ROOT = Ref(true)


@inline mpi_comm() = MPI.COMM_WORLD

@inline mpi_rank() = MPI_RANK[]

@inline mpi_nranks() = MPI_SIZE[]

@inline mpi_isparallel() = MPI_IS_PARALLEL[]

# This is not type-stable but that's okay since we want to get rid of it anyway
# and it's not used in performance-critical parts. The alternative we used before,
# calling something like `eval(:(mpi_parallel() = True()))` in `init_mpi()`,
# causes invalidations and slows down the first call to Trixi.jl.
function mpi_parallel()
  if mpi_isparallel()
    return True()
  else
    return False()
  end
end

@inline mpi_isroot() = MPI_IS_ROOT[]

@inline mpi_root() = 0

@inline function mpi_println(args...)
  if mpi_isroot()
    println(args...)
  end
  return nothing
end
@inline function mpi_print(args...)
  if mpi_isroot()
    print(args...)
  end
  return nothing
end


"""
    ode_norm(u, t)

Implementation of the weighted L2 norm of Hairer and Wanner used for error-based
step size control in OrdinaryDiffEq.jl. This function is aware of MPI and uses
global MPI communication when running in parallel.

You must pass this function as a keyword argument
`internalnorm=ode_norm`
to OrdinaryDiffEq.jl's `solve` when using error-based step size control with MPI
parallel execution of Trixi.jl.

See the "Advanced Adaptive Stepsize Control" section of the [documentation](https://docs.sciml.ai/DiffEqDocs/stable/basics/common_solver_opts/)
"""
ode_norm(u::Number, t) = @fastmath abs(u)
function ode_norm(u::AbstractArray, t)
  local_sumabs2 = recursive_sum_abs2(u) # sum(abs2, u)
  local_length  = recursive_length(u)   # length(u)
  if mpi_isparallel()
    global_sumabs2, global_length = MPI.Allreduce([local_sumabs2, local_length], +, mpi_comm())
    return sqrt(global_sumabs2 / global_length)
  else
    return sqrt(local_sumabs2 / local_length)
  end
end

# Recursive `sum(abs2, ...)` and `length(...)` are required when dealing with
# arrays of arrays, e.g., when using `DGMulti` solvers with an array-of-structs
# (`Array{SVector}`) or a structure-of-arrays (`StructArray`). We need to take
# care of these situations when allowing to use `ode_norm` as default norm in
# OrdinaryDiffEq.jl throughout all applications of Trixi.jl.
recursive_sum_abs2(u::Number) = abs2(u)
# Use `mapreduce` instead of `sum` since `sum` from StaticArrays.jl does not
# support the kwarg `init`
# We need `init=zero(eltype(eltype(u))` below to deal with arrays of `SVector`s etc.
# A better solution would be `recursive_unitless_bottom_eltype` from 
# https://github.com/SciML/RecursiveArrayTools.jl
# However, what you have is good enough for us for now, so we don't need this 
# additional dependency at the moment.
recursive_sum_abs2(u::AbstractArray) = mapreduce(recursive_sum_abs2, +, u; init=zero(eltype(eltype(u))))

recursive_length(u::Number) = length(u)
recursive_length(u::AbstractArray{<:Number}) = length(u)
recursive_length(u::AbstractArray{<:AbstractArray}) = sum(recursive_length, u)
function recursive_length(u::AbstractArray{<:StaticArrays.StaticArray{S, <:Number}}) where {S}
  prod(StaticArrays.Size(eltype(u))) * length(u)
end


"""
    ode_unstable_check(dt, u, semi, t)

Implementation of the basic check for instability used in OrdinaryDiffEq.jl.
Instead of checking something like `any(isnan, u)`, this function just checks
`isnan(dt)`. This helps when using MPI parallelization, since no additional
global communication is required and all ranks will return the same result.

You should pass this function as a keyword argument
`unstable_check=ode_unstable_check`
to OrdinaryDiffEq.jl's  `solve` when using error-based step size control with MPI
parallel execution of Trixi.jl.

See the "Miscellaneous" section of the [documentation](https://docs.sciml.ai/DiffEqDocs/stable/basics/common_solver_opts/)
"""
ode_unstable_check(dt, u, semi, t) = isnan(dt)
