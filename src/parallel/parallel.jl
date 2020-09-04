domain_id(comm) = MPI.Comm_rank(comm)
domain_id() = MPI.Comm_rank(MPI.COMM_WORLD)

n_domains(comm) = MPI.Comm_size(comm)
n_domains() = MPI.Comm_size(MPI.COMM_WORLD)

is_parallel(comm) = n_domains(comm) > 1
is_parallel() = is_parallel(MPI.COMM_WORLD)

is_serial(comm) = !is_parallel(comm)
is_serial() = is_serial(MPI.COMM_WORLD)

is_mpi_root(comm) = is_serial() || domain_id(comm) == 0
is_mpi_root() = is_mpi_root(MPI.COMM_WORLD)
