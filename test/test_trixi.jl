using Test: @test_nowarn, @test
import Trixi


"""
    test_trixi_run(parameters_file; l2=nothing, linf=nothing)

Test Trixi by calling `Trixi.run` with `parameters_file` as the parameters
file. By default, only the absence of error output is checked. If
`l2` or `linf` are specified, in addition the resulting L2/Linf errors
are compared approximately against these reference values.
"""
function test_trixi_run(parameters_file; l2=nothing, linf=nothing)
  # Run basic test to ensure that there is no output to STDERR
  @test_nowarn l2_measured, linf_measured, _ = Trixi.run(parameters_file)

  # If present, compare L2 and Linf errors against reference values. Use
  # `collect` to promote both reference and measured values to arrays, such
  # that they can be specified in any collection type. Use dot syntax on
  # `isapprox`, as by default it will compare arrays using a norm and not
  # component-wise.
  if !isnothing(l2)
    @test all(isapprox.(collect(l2), collect(l2_measured)))
  end
  if !isnothing(linf)
    @test all(isapprox.(collect(linf), collect(linf_measured)))
  end
end
