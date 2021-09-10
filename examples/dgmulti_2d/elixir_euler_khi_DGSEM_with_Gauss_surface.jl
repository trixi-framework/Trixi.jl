using Trixi, OrdinaryDiffEq

polydeg = 3

# use DGSEM quadrature in the volume
rq1D, wq1D = StartUpDG.gauss_lobatto_quad(0, 0, polydeg)
rq, sq = vec.(StartUpDG.NodesAndModes.meshgrid(rq1D))
wr, ws = vec.(StartUpDG.NodesAndModes.meshgrid(wq1D))
wq = wr .* ws

r1D, w1D = StartUpDG.gauss_quad(0, 0, polydeg) # Gauss face

# # Clenshaw-Curtis quadrature
# n = polydeg + 1 # number of Chebyshev nodes
# r1D = sort(@. cos((2*(1:n)-1)/(2*n)*pi)) # first kind
# # r1D = sort(@. -cos((0:polydeg)*pi/polydeg)) # 2nd kind
# Vq_approx = StartUpDG.vandermonde(StartUpDG.Line(), length(r1D)-1, r1D)
# w1D = Vq_approx' \ [sqrt(2); zeros(size(Vq_approx, 2)-1)]

surface_flux = FluxLaxFriedrichs()
# surface_flux = flux_hll
dg = DGMulti(polydeg=polydeg, element_type = Quad(), approximation_type = Polynomial(),
             surface_integral = SurfaceIntegralWeakForm(surface_flux),
             volume_integral = VolumeIntegralFluxDifferencing(flux_ranocha),
             quad_rule_vol = (rq, sq, wq), quad_rule_face = (r1D, w1D))

# dg = DGMulti(polydeg=polydeg, element_type = Quad(), approximation_type = SBP(),
#              surface_integral = SurfaceIntegralWeakForm(surface_flux),
#              volume_integral = VolumeIntegralFluxDifferencing(flux_ranocha))

equations = CompressibleEulerEquations2D(1.4)

function initial_condition_kelvin_helmholtz_instability(x, t, equations::CompressibleEulerEquations2D)
  # change discontinuity to tanh
  # typical resolution 128^2, 256^2
  # domain size is [-1,+1]^2
  slope = 15
   # discontinuous function with value 2 for -.5 <= x <= .5 and 0 elsewhere
  B = tanh(slope * x[2] + 7.5) - tanh(slope * x[2] - 7.5)
  rho = 0.5 + 0.75 * B
  v1 = 0.5 * (B - 1)
  # # scale by (1 + .01 * sin(pi*x)*sin(pi*y)) to break symmetry
  # v2 = 0.1 * sin(2 * pi * x[1]) * (1.0 + 1e-2 * sin(pi * x[1]) * sin(pi * x[2]))
  v2 = 0.1 * sin(2 * pi * x[1])
  p = 1.0
  return prim2cons(SVector(rho, v1, v2, p), equations)
end
initial_condition = initial_condition_kelvin_helmholtz_instability

num_cells_per_dimension = 64
vertex_coordinates, EToV = StartUpDG.uniform_mesh(dg.basis.elementType, num_cells_per_dimension)
mesh = VertexMappedMesh(vertex_coordinates, EToV, dg, is_periodic=(true, true))
semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, dg)

tspan = (0.0, 10)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()
alive_callback = AliveCallback(alive_interval=100)
analysis_interval = 500
analysis_callback = AnalysisCallback(semi, interval=analysis_interval, uEltype=real(dg))
callbacks = CallbackSet(summary_callback,
                        analysis_callback,
                        alive_callback)

###############################################################################
# run the simulation

# sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false),
#             dt = estimate_dt(mesh, dg), save_everystep=false, callback=callbacks);

tol = 1.0e-8
tsave = LinRange(tspan..., 50)
sol = solve(ode, RDPK3SpFSAL49(), abstol=tol, reltol=tol,
            save_everystep=false, saveat=tsave, callback=callbacks);

summary_callback() # print the timer summary

