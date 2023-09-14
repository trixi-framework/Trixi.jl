module TestAqua

using Aqua
using Test
using Trixi

include("test_trixi.jl")

@timed_testset "Aqua.jl" begin
    Aqua.test_all(Trixi,
                  ambiguities = false,
                  piracy = (treat_as_own = [Trixi.StartUpDG.RefElemData,
                                            Trixi.StartUpDG.MeshData],))
end

end #module
