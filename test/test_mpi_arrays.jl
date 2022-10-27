module TestTrixiMPIArrays

using Test
using LinearAlgebra: norm
using Trixi: TrixiMPIArray, mpi_isroot, mpi_nranks, ode_norm, ode_unstable_check


@testset "TestTrixiMPIArrays" begin

@testset "show" begin
  u_parent = zeros(5)
  u_mpi = TrixiMPIArray(u_parent)

  # output on all ranks
  @test_nowarn show(stdout, u_mpi)
  @test_nowarn show(stdout, MIME"text/plain"(), u_mpi)

  # test whether show works also on a single rank
  if mpi_isroot()
    @test_nowarn show(stdout, u_mpi)
    @test_nowarn show(stdout, MIME"text/plain"(), u_mpi)
  end
end


@testset "vector interface" begin
  u_parent = ones(5)
  u_mpi = TrixiMPIArray(u_parent)

  @test sum(u_mpi) ≈ 5 * mpi_nranks()

  @test_nowarn resize!(u_mpi, 4)
  @test sum(u_mpi) ≈ 4 * mpi_nranks()
  @test sum(u_parent) == 4

  u_mpi2 = copy(u_mpi)
  @test sum(u_mpi2) ≈ sum(u_mpi)

  res = @. u_mpi / u_mpi
  @test res isa TrixiMPIArray

  res = @. u_mpi / u_parent
  @test res isa TrixiMPIArray

  res = @. 5 * u_mpi
  @test res isa TrixiMPIArray

  @test strides(u_mpi) == strides(u_parent)
  @test Base.elsize(u_mpi) == sizeof(Float64)
end


@testset "ODE interface" begin
  u_parent = [1.0, -2.0, 3.5, -4.0]
  u_mpi = TrixiMPIArray(u_parent)

  # duplicating a vector doesn't change the norm weighted by the global length
  @test ode_norm(u_parent, 0.0) ≈ norm(u_parent) / sqrt(length(u_parent))
  @test ode_norm(u_mpi, 0.0) ≈ ode_norm(u_parent, 0.0)

  u_parent[1] = NaN
  @test ode_unstable_check(rand(), u_mpi, nothing, 0.0) == false
  @test ode_unstable_check(NaN, u_mpi, nothing, 0.0) == true
end


end # TestTrixiMPIArrays

end # module
