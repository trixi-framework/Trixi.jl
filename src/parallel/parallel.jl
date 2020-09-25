"""
    init_mpi

Initialize MPI by calling `MPI.Initialized()`. The function will check if MPI is already initialized
and if yes, do nothing, thus it is safe to call it multiple times.
"""
function init_mpi()
  if !MPI.Initialized()
    # MPI.THREAD_FUNNELED: Only main thread makes MPI calls
    provided = MPI.Init_thread(MPI.THREAD_FUNNELED)
    @assert provided >= MPI.THREAD_FUNNELED "MPI library with insufficient threading support"
  end

  return nothing
end


const MPI_RANK = Ref(-1)
const MPI_SIZE = Ref(-1)
const MPI_IS_PARALLEL = Ref(false)
const MPI_IS_SERIAL = Ref(true)
const MPI_IS_ROOT = Ref(true)


@inline mpi_comm() = MPI.COMM_WORLD

@inline domain_id(comm) = MPI.Comm_rank(comm)
@inline domain_id() = MPI_RANK[]

@inline n_domains(comm) = MPI.Comm_size(comm)
@inline n_domains() = MPI_SIZE[]

@inline is_parallel(comm) = n_domains(comm) > 1
@inline is_parallel() = MPI_IS_PARALLEL[]

@inline is_serial(comm) = !is_parallel(comm)
@inline is_serial() = MPI_IS_SERIAL[]

@inline is_mpi_root(comm) = is_serial() || domain_id(comm) == 0
@inline is_mpi_root() = MPI_IS_ROOT[]

@inline mpi_root() = 0

@inline mpi_println(args...) = is_mpi_root() && println(args...)
@inline mpi_print(args...) = is_mpi_root() && print(args...)
