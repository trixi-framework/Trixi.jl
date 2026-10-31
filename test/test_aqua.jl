# Load the packages triggering the package extensions such that the checks below cover
# them as well. The GPU extensions are not included since their packages may not be
# loadable on the CI runners.
@testsnippet TrixiExtensions begin
    using CairoMakie, Convex, ECOS, NLsolve, OrdinaryDiffEqCore, Plots,
          SparseConnectivityTracer
    extensions = [Base.get_extension(Trixi, name)
                  for name in (:TrixiConvexECOSExt, :TrixiMakieExt, :TrixiNLsolveExt,
                               :TrixiOrdinaryDiffEqCoreExt, :TrixiPlotsExt,
                               :TrixiSparseConnectivityTracerExt)]
end

@testitem "Aqua.jl" setup=[Setup] tags=[:qa] begin
    using Aqua

    Aqua.test_all(Trixi,
                  ambiguities = false,
                  unbound_args = false, # FIXME: UnstructuredSortedBoundaryTypes
                  # exceptions necessary for adding a new method `StartUpDG.estimate_h`
                  # in src/solvers/dgmulti/sbp.jl
                  piracies = (treat_as_own = [Trixi.StartUpDG.RefElemData,
                                  Trixi.StartUpDG.MeshData],),
                  # exception necessary because StableRNGs.jl and ADTypes.jl are only used in an extension
                  stale_deps = (ignore = [:StableRNGs, :ADTypes],))
end

@testitem "ExplicitImports.jl" setup=[Setup, TrixiExtensions] tags=[:qa] begin
    using ExplicitImports: check_no_implicit_imports, check_no_stale_explicit_imports

    # The checks of `Trixi` include its loaded extensions.
    @test all(!isnothing, extensions)

    @test isnothing(check_no_implicit_imports(Trixi,
                                              skip = (Core, Base, Trixi.P4est, Trixi.T8code,
                                                      Trixi.EllipsisNotation)))
    @test isnothing(check_no_stale_explicit_imports(Trixi,
                                                    ignore = (:derivative_operator,
                                                              :periodic_derivative_operator,
                                                              :upwind_operators,
                                                              :derivative_discontinuity!,
                                                              Symbol("@batch"))))
end

@testitem "Closure boxes" setup=[Setup, TrixiExtensions] tags=[:qa] begin
    # Captured variables that are boxed (`Core.Box`) are untyped, which makes the code
    # using them type-unstable.
    isdefined(Test, :detect_closure_boxes) ||
        error("`Test.detect_closure_boxes` requires Julia v1.14 or newer")

    @test all(!isnothing, extensions)

    @test Test.detect_closure_boxes(Trixi, extensions...) == []
end
