module TestType

using Test
using Trixi

include("test_trixi.jl")

# Start with a clean environment: remove Trixi.jl output directory if it exists
outdir = "out"
isdir(outdir) && rm(outdir, recursive = true)

# Run unit tests for various equations
@testset "Float32 Type Stability" begin
    @timed_testset "Type stability for acoustic perturbation 2D" begin
        v_mean_global = (0.0f0, 0.0f0)
        c_mean_global = 1.0f0
        rho_mean_global = 1.0f0
        equations = AcousticPerturbationEquations2D(v_mean_global, c_mean_global,
                                                    rho_mean_global)

        x = SVector(0.0f0, 0.0f0)
        t = 0.0f0
        u = u_ll = u_rr = SVector(1.0f0, 1.0f0, 1.0f0, 1.0f0, 1.0f0, 1.0f0, 1.0f0)
        orientations = [1, 2]
        normal_direction = SVector(1.0f0, 1.0f0)

        @test eltype(initial_condition_constant(x, t, equations)) == Float32
        @test eltype(initial_condition_convergence_test(x, t, equations)) == Float32
        @test eltype(initial_condition_gauss(x, t, equations)) == Float32

        @test eltype(source_terms_convergence_test(u, x, t, equations)) == Float32
        @test eltype(source_terms_convergence_test(u, x, t, equations)) == Float32

        for orientation in orientations
            @test eltype(flux(u, orientation, equations)) == Float32
        end

        @test eltype(flux(u, normal_direction, equations)) == Float32

        dissipation = DissipationLocalLaxFriedrichs()
        # test orientation
        @test eltype(dissipation(u_ll, u_rr, orientations[1], equations)) == Float32
        # test normal direction
        @test eltype(dissipation(u_ll, u_rr, normal_direction, equations)) == Float32
    end
end

end # module
