
"""
    init_mpi()

Initialize MPI by calling `MPI.Initialized()`. The function will check if MPI is already initialized
and if yes, do nothing, thus it is safe to call it multiple times.
"""
function init_mpi()
  if MPI_INITIALIZED[]
    return nothing
  end

  if MPI.Initialized()
    @assert MPI.Query_thread() >= MPI.THREAD_FUNNELED "MPI already initialized with insufficient threading support"
  else
    # MPI.THREAD_FUNNELED: Only main thread makes MPI calls
    provided = MPI.Init_thread(MPI.THREAD_FUNNELED)
    @assert provided >= MPI.THREAD_FUNNELED "MPI library with insufficient threading support"
  end

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
