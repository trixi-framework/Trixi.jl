# Similar to unstructured_2d_dgsem/elixir_shallowwater_ec_float32.jl
# but using Float32 instead of the default Float64

using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the shallow water equations with a discontinuous
# bottom topography function

equations = ShallowWaterEquations2D(gravity_constant = 9.81f0)

# Note, this initial condition is used to compute errors in the analysis callback but the initialization is
# overwritten by `initial_condition_ec_discontinuous_bottom` below.
initial_condition = initial_condition_weak_blast_wave

###############################################################################
# Get the DG approximation space

volume_flux = (flux_wintermeyer_etal, flux_nonconservative_wintermeyer_etal)
solver = DGSEM(polydeg = 6,
               surface_flux = (flux_fjordholm_etal, flux_nonconservative_fjordholm_etal),
               volume_integral = VolumeIntegralFluxDifferencing(volume_flux),
               RealT = Float32)

###############################################################################
# This setup is for the curved, split form entropy conservation testing (needs periodic BCs)

# Get the unstructured quad mesh from a file (downloads the file if not available locally)
mesh_file = Trixi.download("https://gist.githubusercontent.com/andrewwinters5000/8f8cd23df27fcd494553f2a89f3c1ba4/raw/85e3c8d976bbe57ca3d559d653087b0889535295/mesh_alfven_wave_with_twist_and_flip.mesh",
                           joinpath(@__DIR__, "mesh_alfven_wave_with_twist_and_flip.mesh"))

mesh = UnstructuredMesh2D(mesh_file, periodicity = true, RealT = Float32)

# Create the semi discretization object
semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)

###############################################################################
# ODE solver

tspan = (0.0f0, 2.0f0)
ode = semidiscretize(semi, tspan)

###############################################################################
# Workaround to set a discontinuous bottom topography and initial condition for debugging and testing.

# alternative version of the initial conditinon used to setup a truly discontinuous
# bottom topography function and initial condition for this academic testcase of entropy conservation.
# The errors from the analysis callback are not important but `∑∂S/∂U ⋅ Uₜ` should be around machine roundoff
# In contrast to the usual signature of initial conditions, this one get passed the
# `element_id` explicitly. In particular, this initial conditions works as intended
# only for the specific mesh loaded above!
function initial_condition_ec_discontinuous_bottom(x, t, element_id,
                                                   equations::ShallowWaterEquations2D)
    RealT = eltype(x)

    # Set up polar coordinates
    inicenter = SVector{2, RealT}(0.7, 0.7)
    x_norm = x[1] - inicenter[1]
    y_norm = x[2] - inicenter[2]
    r = sqrt(x_norm^2 + y_norm^2)
    phi = atan(y_norm, x_norm)
    sin_phi, cos_phi = sincos(phi)

    # Set the background values
    H = 3.25f0
    v1 = zero(RealT)
    v2 = zero(RealT)
    b = zero(RealT)

    # setup the discontinuous water height and velocities
    if element_id == 10
        H = 4.0f0
        v1 = convert(RealT, 0.1882) * cos_phi
        v2 = convert(RealT, 0.1882) * sin_phi
    end

    # Setup a discontinuous bottom topography using the element id number
    if element_id == 7
        b = 2 + 0.5f0 * sinpi(2 * x[1]) + 0.5f0 * cospi(2 * x[2])
    end

    return prim2cons(SVector(H, v1, v2, b), equations)
end

# point to the data we want to augment
u = Trixi.wrap_array(ode.u0, semi)
# reset the initial condition
for element in eachelement(semi.solver, semi.cache)
    for j in eachnode(semi.solver), i in eachnode(semi.solver)
        x_node = Trixi.get_node_coords(semi.cache.elements.node_coordinates, equations,
                                       semi.solver, i, j, element)
        u_node = initial_condition_ec_discontinuous_bottom(x_node, first(tspan), element,
                                                           equations)
        Trixi.set_node_vars!(u, u_node, equations, semi.solver, i, j, element)
    end
end

###############################################################################
# Callbacks

summary_callback = SummaryCallback()

analysis_interval = 1000
analysis_callback = AnalysisCallback(semi, interval = analysis_interval)

alive_callback = AliveCallback(analysis_interval = analysis_interval)

save_solution = SaveSolutionCallback(interval = 1000,
                                     save_initial_solution = true,
                                     save_final_solution = true)

stepsize_callback = StepsizeCallback(cfl = 1.0f0)

callbacks = CallbackSet(summary_callback, analysis_callback, alive_callback, save_solution,
                        stepsize_callback)

###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false),
            dt = 1.0f0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep = false, callback = callbacks);
summary_callback() # print the timer summary
