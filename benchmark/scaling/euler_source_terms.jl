# Compressible Euler equations on a 3D P4estMesh with about 2.1 million degrees of freedom.

elixir = joinpath(examples_dir(), "p4est_3d_dgsem", "elixir_euler_source_terms.jl")
# The steps grow with the ranks and threads, so every point measures a similar wall time
# under ideal scaling.
overrides = (; initial_refinement_level = 3,
             maxiters = 16 * Trixi.mpi_nranks() * Threads.nthreads(),
             analysis_callback = TrivialCallback(), save_solution = TrivialCallback())
