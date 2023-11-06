module TestExamples2DEulerMulticomponent

using Test
using Trixi

include("test_trixi.jl")

# pathof(Trixi) returns /path/to/Trixi/src/Trixi.jl, dirname gives the parent directory
EXAMPLES_DIR = joinpath(pathof(Trixi) |> dirname |> dirname, "examples", "tree_2d_dgsem")

@testset "MHD Multi-ion" begin
    @trixi_testset "elixir_mhdmultiion_ec.jl" begin
        @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhdmultiion_ec.jl"),
                            l2=[1.56133690e-02, 1.56211211e-02, 2.44289260e-02,
                                1.17053210e-02, 1.35748661e-01,
                                1.35779534e-01, 1.34646112e-01, 1.34813656e-01,
                                1.93724876e-02, 2.70357315e-01,
                                2.70356924e-01, 2.69252524e-01, 1.86315505e-01],
                            linf=[1.06156769e-01, 1.15019769e-01, 1.32816030e-01,
                                7.65402322e-02, 2.45518940e-01,
                                2.46123607e-01, 1.82733442e-01, 4.24743430e-01,
                                1.27620999e-01, 4.58874938e-01,
                                4.65364246e-01, 3.56983044e-01, 3.94035665e-01])
    end

    @trixi_testset "elixir_mhdmultiion_rotor.jl tspan = (0., 0.001)" begin
        @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhdmultiion_rotor.jl"),
                            l2=[9.10689060e-03, 1.57109974e-02, 5.47502000e-06,
                                4.33887866e-02, 6.85503869e-02,
                                6.44021766e-02, 2.79487163e-03, 7.85539922e-02,
                                4.33883209e-02, 6.85496075e-02,
                                6.44066193e-02, 5.58969701e-03, 7.85504216e-02],
                            linf=[1.47204796e-01, 2.33759231e-01, 2.89189051e-05,
                                1.06452623e+00, 3.36709456e+00,
                                2.93566426e+00, 1.53123364e-02, 3.99872907e+00,
                                1.06455108e+00, 3.36725655e+00,
                                2.93570704e+00, 3.05339471e-02, 3.99892281e+00],
                            tspan=(0.0, 0.001))
    end
end

end # module
