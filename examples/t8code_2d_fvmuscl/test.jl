using MPI

MPI.Init()

comm = MPI.COMM_WORLD

println("Hello, I'm $(MPI.Comm_rank(comm)) of $(MPI.Comm_size(comm)).")

MPI.Finalize()