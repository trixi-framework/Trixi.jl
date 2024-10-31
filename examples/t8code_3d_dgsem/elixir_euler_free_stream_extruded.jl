using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the compressible Euler equations

equations = CompressibleEulerEquations3D(1.4)

initial_condition = initial_condition_constant

boundary_conditions = Dict(:all => BoundaryConditionDirichlet(initial_condition))

solver = DGSEM(polydeg = 3, surface_flux = flux_lax_friedrichs,
               volume_integral = VolumeIntegralWeakForm())

# Mapping as described in https://arxiv.org/abs/2012.12040 but reduced to 2D.
# This particular mesh is unstructured in the yz-plane, but extruded in x-direction.
# Apply the warping mapping in the yz-plane to get a curved 2D mesh that is extruded
# in x-direction to ensure free stream preservation on a non-conforming mesh.
# See https://doi.org/10.1007/s10915-018-00897-9, Section 6.
function mapping(xi, eta_, zeta_)
    # Transform input variables between -1 and 1 onto [0,3]
    eta = 1.5 * eta_ + 1.5
    zeta = 1.5 * zeta_ + 1.5

    z = zeta +
        1 / 6 * (cos(1.5 * pi * (2 * eta - 3) / 3) *
                 cos(0.5 * pi * (2 * zeta - 3) / 3))

    y = eta + 1 / 6 * (cos(0.5 * pi * (2 * eta - 3) / 3) *
                       cos(2 * pi * (2 * z - 3) / 3))

    return SVector(xi, y, z)
end

# Unstructured mesh with 48 cells of the cube domain [-1, 1]^3
mesh_file = Trixi.download("https://gist.githubusercontent.com/efaulhaber/b8df0033798e4926dec515fc045e8c2c/raw/b9254cde1d1fb64b6acc8416bc5ccdd77a240227/cube_unstructured_2.inp",
                           joinpath(@__DIR__, "cube_unstructured_2.inp"))

mesh = T8codeMesh(mesh_file, 3; polydeg = 3,
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

# Refine quadrants in y-direction of each tree at one edge to level 2
function adapt_callback(forest, ltreeid, eclass_scheme, lelemntid, elements, is_family,
                        user_data)
    el = unsafe_load(Ptr{t8_dhex_t}(elements[1]))

    if convert(Int, ltreeid) < 4 && el.x == 0 && el.y == 0 && el.level < 2
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

sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false), #maxiters=1,
            dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep = false, callback = callbacks);
summary_callback() # print the timer summary

# Finalize `T8codeMesh` to make sure MPI related objects in t8code are
# released before `MPI` finalizes.
!isinteractive() && finalize(mesh)
