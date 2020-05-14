using Test: @test_nowarn, @test
import Trixi


"""
    test_trixi_run(parameters_file; l2=nothing, linf=nothing, rtol=0.001)

Test Trixi by calling `Trixi.run` with `parameters_file` as the parameters
file. By default, only the absence of error output is checked. If
`l2` or `linf` are specified, in addition the resulting L2/Linf errors
are compared approximately against these reference values, using `rtol` for
the relative tolerance.
"""
function test_trixi_run(parameters_file; l2=nothing, linf=nothing, rtol=0.001)
  # Run basic test to ensure that there is no output to STDERR
  @test_nowarn l2_measured, linf_measured, _ = Trixi.run(parameters_file)

  # If present, compare L2 and Linf errors against reference values
  if !isnothing(l2)
    @test all(isapprox(a, b, rtol=rtol) for (a, b) in zip(l2, l2_measured))
  end
  if !isnothing(linf)
    @test all(isapprox(a, b, rtol=rtol) for (a, b) in zip(linf, linf_measured))
  end
end
