module TestAqua

using Aqua
using ExplicitImports: check_no_implicit_imports, check_no_stale_explicit_imports
using Test
using Trixi

include("test_trixi.jl")

@timed_testset "Aqua.jl" begin
    Aqua.test_all(Trixi,
                  ambiguities = false,
                  # exceptions necessary for adding a new method `StartUpDG.estimate_h`
                  # in src/solvers/dgmulti/sbp.jl
                  piracies = (treat_as_own = [Trixi.StartUpDG.RefElemData,
                                  Trixi.StartUpDG.MeshData],))
    @test isnothing(check_no_implicit_imports(Trixi,
                                              skip = (Core, Base, Trixi.P4est, Trixi.T8code,
                                                      Trixi.EllipsisNotation)))
    @test isnothing(check_no_stale_explicit_imports(Trixi,
                                                    ignore = (:derivative_operator,
                                                              :periodic_derivative_operator,
                                                              :upwind_operators,
                                                              Symbol("@batch"))))
end

end #module
