# The MPI example tests live in the `test_mpi_*.jl` files and are tagged `:mpi`.
# They are discovered automatically by `@run_package_tests` and run inside the
# `mpiexec`-launched worker process set up in `runtests.jl`. Items that fail
# often on Windows CI are additionally tagged `:mpi_skip_windows`.

@testitem "MPI supporting functionality" setup=[Setup] tags=[:mpi] begin
    using Trixi: Trixi, ode_norm, SVector
    t = 0.5
    let u = 1.0
        @test ode_norm(u, t) ≈ Trixi.DiffEqBase.ODE_DEFAULT_NORM(u, t)
    end
    let u = [1.0, -2.0]
        @test ode_norm(u, t) ≈ Trixi.DiffEqBase.ODE_DEFAULT_NORM(u, t)
    end
    let u = [SVector(1.0, -2.0), SVector(0.5, -0.1)]
        @test ode_norm(u, t) ≈ Trixi.DiffEqBase.ODE_DEFAULT_NORM(u, t)
    end
end
