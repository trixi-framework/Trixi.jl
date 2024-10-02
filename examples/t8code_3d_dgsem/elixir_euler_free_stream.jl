using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the compressible Euler equations

equations = CompressibleEulerEquations3D(1.4)

initial_condition = initial_condition_constant

boundary_conditions = Dict(:all => BoundaryConditionDirichlet(initial_condition))

# Solver with polydeg=4 to ensure free stream preservation (FSP) on non-conforming meshes.
# The polydeg of the solver must be at least twice as big as the polydeg of the mesh.
# See https://doi.org/10.1007/s10915-018-00897-9, Section 6.
solver = DGSEM(polydeg = 4, surface_flux = flux_lax_friedrichs,
               volume_integral = VolumeIntegralWeakForm())

# Mapping as described in https://arxiv.org/abs/2012.12040 but with less warping.
# The mapping will be interpolated at tree level, and then refined without changing
# the geometry interpolant. This can yield problematic geometries if the unrefined mesh
# is not fine enough.
function mapping(xi_, eta_, zeta_)
    # Transform input variables between -1 and 1 onto [0,3]
    xi = 1.5 * xi_ + 1.5
    eta = 1.5 * eta_ + 1.5
    zeta = 1.5 * zeta_ + 1.5

    y = eta +
        1 / 6 * (cos(1.5 * pi * (2 * xi - 3) / 3) *
         cos(0.5 * pi * (2 * eta - 3) / 3) *
         cos(0.5 * pi * (2 * zeta - 3) / 3))

    x = xi +
        1 / 6 * (cos(0.5 * pi * (2 * xi - 3) / 3) *
         cos(2 * pi * (2 * y - 3) / 3) *
         cos(0.5 * pi * (2 * zeta - 3) / 3))

    z = zeta +
        1 / 6 * (cos(0.5 * pi * (2 * x - 3) / 3) *
         cos(pi * (2 * y - 3) / 3) *
         cos(0.5 * pi * (2 * zeta - 3) / 3))

    return SVector(x, y, z)
end

# Unstructured mesh with 68 cells of the cube domain [-1, 1]^3
mesh_file = Trixi.download("https://gist.githubusercontent.com/efaulhaber/d45c8ac1e248618885fa7cc31a50ab40/raw/37fba24890ab37cfa49c39eae98b44faf4502882/cube_unstructured_1.inp",
                           joinpath(@__DIR__, "cube_unstructured_1.inp"))

mesh = T8codeMesh(mesh_file, 3; polydeg = 2,
                  mapping = mapping,
                  initial_refinement_level = 0)

# Note: This is actually a `p8est_quadrant_t` which is much bigger than the
# following struct. But we only need the first four fields for our purpose.
struct t8_dhex_t
    x::Int32
    y::Int32
    z::Int32
    level::Int8
    # [...] # See `p8est.h` in `p4est` for more info.
end

# Refine bottom left quadrant of each second tree to level 2
function adapt_callback(forest, ltreeid, eclass_scheme, lelemntid, elements, is_family,
                        user_data)
    el = unsafe_load(Ptr{t8_dhex_t}(elements[1]))

    if iseven(convert(Int, ltreeid)) && el.x == 0 && el.y == 0 && el.z == 0 &&
       el.level < 2
        # return true (refine)
        return 1
    else
        # return false (don't refine)
        return 0
    end
end

Trixi.adapt!(mesh, adapt_callback)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                    boundary_conditions = boundary_conditions)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 1.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval = analysis_interval)

alive_callback = AliveCallback(analysis_interval = analysis_interval)

stepsize_callback = StepsizeCallback(cfl = 1.2)

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
