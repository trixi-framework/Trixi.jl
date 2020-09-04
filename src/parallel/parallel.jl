@inline mpi_comm() = MPI.COMM_WORLD

@inline domain_id(comm) = MPI.Comm_rank(comm)
@inline domain_id() = MPI.Comm_rank(mpi_comm())

@inline n_domains(comm) = MPI.Comm_size(comm)
@inline n_domains() = MPI.Comm_size(mpi_comm())

@inline is_parallel(comm) = n_domains(comm) > 1
@inline is_parallel() = is_parallel(mpi_comm())

@inline is_serial(comm) = !is_parallel(comm)
@inline is_serial() = is_serial(mpi_comm())

@inline is_mpi_root(comm) = is_serial() || domain_id(comm) == 0
@inline is_mpi_root() = is_mpi_root(mpi_comm())

@inline mpi_root() = 0
