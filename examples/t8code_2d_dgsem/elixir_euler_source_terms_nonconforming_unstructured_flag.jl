using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the compressible Euler equations

equations = CompressibleEulerEquations2D(1.4)

initial_condition = initial_condition_convergence_test

source_terms = source_terms_convergence_test

# BCs must be passed as Dict
boundary_condition = BoundaryConditionDirichlet(initial_condition)
boundary_conditions = Dict(:all => boundary_condition)

solver = DGSEM(polydeg = 3, surface_flux = flux_lax_friedrichs)

# Deformed rectangle that looks like a waving flag,
# lower and upper faces are sinus curves, left and right are vertical lines.
f1(s) = SVector(-1.0, s - 1.0)
f2(s) = SVector(1.0, s + 1.0)
f3(s) = SVector(s, -1.0 + sin(0.5 * pi * s))
f4(s) = SVector(s, 1.0 + sin(0.5 * pi * s))
faces = (f1, f2, f3, f4)

Trixi.validate_faces(faces)
mapping_flag = Trixi.transfinite_mapping(faces)

# Get the uncurved mesh from a file (downloads the file if not available locally)
# Unstructured mesh with 24 cells of the square domain [-1, 1]^n
mesh_file = Trixi.download("https://gist.githubusercontent.com/efaulhaber/63ff2ea224409e55ee8423b3a33e316a/raw/7db58af7446d1479753ae718930741c47a3b79b7/square_unstructured_2.inp",
                           joinpath(@__DIR__, "square_unstructured_2.inp"))

mesh = T8codeMesh(mesh_file, 2; polydeg = 3,
                  mapping = mapping_flag,
                  initial_refinement_level = 1)

function adapt_callback(forest, ltreeid, eclass_scheme, lelemntid, elements, is_family,
                        user_data)
    vertex = Vector{Cdouble}(undef, 3)

    Trixi.t8_element_vertex_reference_coords(eclass_scheme, elements[1], 0, pointer(vertex))

    level = Trixi.t8_element_level(eclass_scheme, elements[1])

    # TODO: Make this condition more general.
    if vertex[1] < 1e-8 && vertex[2] < 1e-8 && level < 2
        # return true (refine)
        return 1
    else
        # return false (don't refine)
        return 0
    end
end

Trixi.adapt!(mesh, adapt_callback)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                    source_terms = source_terms,
                                    boundary_conditions = boundary_conditions)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 1.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval = analysis_interval)

alive_callback = AliveCallback(analysis_interval = analysis_interval)

stepsize_callback = StepsizeCallback(cfl = 0.8)

callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback,
                        stepsize_callback)
###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false),
            dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep = false, callback = callbacks);
summary_callback() # print the timer summary

# Finalize `T8codeMesh` to make sure MPI related objects in t8code are
# released before `MPI` finalizes.
!isinteractive() && finalize(mesh)
