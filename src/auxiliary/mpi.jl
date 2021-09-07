# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin


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
# calling something like `eval(:(mpi_parallel() = Val(true)))` in `init_mpi()`,
# causes invalidations and slows down the first call to Trixi.
mpi_parallel()::Union{Val{true}, Val{false}} = Val(mpi_isparallel())

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


end # @muladd
