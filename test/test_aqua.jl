module TestAqua

using Aqua
using Test
using Trixi

include("test_trixi.jl")

@timed_testset "Aqua.jl" begin
    Aqua.test_all(Trixi)
end

end #module
