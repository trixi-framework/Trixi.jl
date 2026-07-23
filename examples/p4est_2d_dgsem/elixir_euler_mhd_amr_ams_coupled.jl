using OrdinaryDiffEqSSPRK, OrdinaryDiffEqLowStorageRK
using Trixi
using P4est

# ── Physics ───────────────────────────────────────────────────────────────────

function initial_condition_mhd(x, t, equations::IdealGlmMhdEquations2D)
    rho = 1.0; v1 = 0.2; v2 = 0.1; v3 = 0.0
    p   = rho^equations.gamma
    r   = sqrt(x[1]^2 + x[2]^2)
    B1  =  x[2] * exp(-10r^2)
    B2  = -x[1] * exp(-10r^2)
    return prim2cons(SVector(rho, v1, v2, v3, p, B1, B2, 0.0, 0.0), equations)
end

function initial_condition_euler(x, t, equations::CompressibleEulerEquations2D)
    rho = 1.0; v1 = 0.2; v2 = 0.1
    return prim2cons(SVector(rho, v1, v2, rho^equations.gamma), equations)
end

equations_mhd   = IdealGlmMhdEquations2D(5/3)
equations_euler = CompressibleEulerEquations2D(5/3)

solver_mhd = DGSEM(polydeg = 3,
                   surface_flux = (flux_lax_friedrichs, flux_nonconservative_powell),
                   volume_integral = VolumeIntegralFluxDifferencing(
                       (flux_hindenlang_gassner, flux_nonconservative_powell)))
solver_euler = DGSEM(polydeg = 3, surface_flux = flux_hll,
                     volume_integral = VolumeIntegralWeakForm())

# Coupling functions: [i,j] maps system-j variables into system-i space.
# System 1 = MHD (9 vars), system 2 = Euler (4 vars).
const coupling_functions = Array{Function}(undef, 2, 2)
coupling_functions[1, 1] = (x, u, eq_other, eq_own) -> u
coupling_functions[1, 2] = (x, u, eq_other, eq_own) ->
    SVector(u[1], u[2], u[3], 0.0, u[4], 0.0, 0.0, 0.0, 0.0)
coupling_functions[2, 1] = (x, u, eq_other, eq_own) -> SVector(u[1], u[2], u[3], u[5])
coupling_functions[2, 2] = (x, u, eq_other, eq_own) -> u

# ── Mesh ──────────────────────────────────────────────────────────────────────

parent_mesh = P4estMesh((32, 32), polydeg = 3,
                        coordinates_min = (-3.0, -3.0),
                        coordinates_max = ( 3.0,  3.0),
                        initial_refinement_level = 0,
                        periodicity = true)

# ── AMR / AMS parameters ──────────────────────────────────────────────────────

const _amr_threshold  = 0.05  # |B|/|B|_max above this → refine to _amr_max_level
const _amr_base_level = 0
const _amr_max_level  = 2
const _ams_low_b2  = 2e-7    # B² threshold: keep cell in MHD region
const _ams_high_b2 = 2e-5    # B² threshold: expand MHD to neighbouring cells

# L2-projection adaptor for AMR solution interpolation.
# AdaptorL2 only needs the DG polynomial basis — no mesh geometry required.
const _adaptor = Trixi.AdaptorL2(solver_mhd.basis)

# Cumulative accepted-step count, kept consistent across AMS restarts so that
# SaveSolutionCallback numbers files continuously.
const _global_naccept = Ref{Int}(0)

# Build a full MHD semidiscretization on the parent mesh.
# Used transiently: for the initial AMS split and during each AMS/AMR callback.
# The returned object is a temporary workspace — callers let it be GC'd immediately.
function _make_parent_semi()
    return SemidiscretizationHyperbolic(
        parent_mesh, equations_mhd, initial_condition_mhd, solver_mhd,
        boundary_conditions = boundary_condition_periodic)
end

# ── Helper functions ──────────────────────────────────────────────────────────

# |B|/|B|_max per parent-mesh cell (used for the AMR indicator).
function compute_alpha_mhd(u_par_ode, n_nodes, n_cells)
    u = reshape(u_par_ode, 9, n_nodes, n_nodes, n_cells)
    B2_max = zero(eltype(u_par_ode))
    for e in 1:n_cells, j in 1:n_nodes, i in 1:n_nodes
        B2_max = max(B2_max, u[6,i,j,e]^2 + u[7,i,j,e]^2)
    end
    B_max = sqrt(max(B2_max, eps(eltype(u_par_ode))))
    alpha = zeros(n_cells)
    for e in 1:n_cells
        Be = zero(eltype(u_par_ode))
        for j in 1:n_nodes, i in 1:n_nodes
            Be = max(Be, sqrt(u[6,i,j,e]^2 + u[7,i,j,e]^2))
        end
        alpha[e] = Be / B_max
    end
    return alpha
end

# AMS region assignment: field-strength threshold + one-layer neighbour buffer.
# Cells with B² > _ams_low_b2 are MHD; neighbours of cells with B² > _ams_high_b2
# are also pulled into MHD. This ensures cells the ring is about to enter are
# included even when they are currently Euler (B = 0 there). Connectivity comes
# from the parent cache so the buffer is correct on any AMR topology.
function ams_split_with_buffer(u_par_ode, n_nodes, n_cells, cache)
    u = reshape(u_par_ode, 9, n_nodes, n_nodes, n_cells)
    b2_max = zeros(n_cells)
    for e in 1:n_cells, j in 1:n_nodes, i in 1:n_nodes
        b2_max[e] = max(b2_max[e], u[6,i,j,e]^2 + u[7,i,j,e]^2)
    end

    is_mhd = b2_max .> _ams_low_b2

    for k in 1:size(cache.interfaces.neighbor_ids, 2)
        e1 = cache.interfaces.neighbor_ids[1, k]
        e2 = cache.interfaces.neighbor_ids[2, k]
        if b2_max[e1] > _ams_high_b2; is_mhd[e2] = true; end
        if b2_max[e2] > _ams_high_b2; is_mhd[e1] = true; end
    end

    n_sides = size(cache.mortars.neighbor_ids, 1) - 1
    for k in 1:size(cache.mortars.neighbor_ids, 2)
        e_c = cache.mortars.neighbor_ids[end, k]
        for s in 1:n_sides
            e_f = cache.mortars.neighbor_ids[s, k]
            if b2_max[e_f] > _ams_high_b2; is_mhd[e_c] = true; end
            if b2_max[e_c] > _ams_high_b2; is_mhd[e_f] = true; end
        end
    end

    return findall(is_mhd), findall(.!is_mhd)
end

# Refine/coarsen flags from alpha and current element levels.
function make_lambda(alpha, current_levels)
    lambda = zeros(Int, length(alpha))
    for e in eachindex(alpha)
        target = alpha[e] > _amr_threshold ? _amr_max_level : _amr_base_level
        lambda[e] = current_levels[e] < target ?  1 :
                    current_levels[e] > target ? -1 : 0
    end
    return lambda
end

# Reconstruct the full parent-mesh MHD state from the two coupled sub-arrays.
# Euler cells contribute (ρ, ρv, E) and get B = ψ = 0.
function reconstruct_parent_mhd(u_ode, u_indices, cell_ids_mhd, cell_ids_euler,
                                 n_nodes, n_parent)
    u_mhd   = reshape(u_ode[u_indices[1]], 9, n_nodes, n_nodes, length(cell_ids_mhd))
    u_euler = reshape(u_ode[u_indices[2]], 4, n_nodes, n_nodes, length(cell_ids_euler))
    u_par   = zeros(9, n_nodes, n_nodes, n_parent)
    for (i, k) in enumerate(cell_ids_mhd)
        u_par[:, :, :, k] .= u_mhd[:, :, :, i]
    end
    for (i, k) in enumerate(cell_ids_euler)
        u_par[1, :, :, k] .= u_euler[1, :, :, i]
        u_par[2, :, :, k] .= u_euler[2, :, :, i]
        u_par[3, :, :, k] .= u_euler[3, :, :, i]
        u_par[5, :, :, k] .= u_euler[4, :, :, i]
    end
    return vec(u_par)
end

# Discover which coupling-boundary names actually exist in a mesh view and build
# the matching NamedTuple. Required because the ring shape means not all four
# cardinal names are present in every view.
function view_boundary_conditions(mesh_view, equations, solver, coupled_bc)
    cache_temp = Trixi.create_cache(mesh_view, equations, solver, nothing, Float64)
    names = unique(cache_temp.boundaries.name)
    isempty(names) && return (;)
    sorted = Tuple(sort(collect(names)))
    return NamedTuple{sorted}(Tuple(coupled_bc for _ in sorted))
end

function build_coupled_semi(cell_ids_mhd, cell_ids_euler)
    mesh_mhd   = P4estMeshView(parent_mesh, cell_ids_mhd)
    mesh_euler = P4estMeshView(parent_mesh, cell_ids_euler)
    coupled_bc = BoundaryConditionCoupledP4est(coupling_functions)
    semi_mhd = SemidiscretizationHyperbolic(
        mesh_mhd, equations_mhd, initial_condition_mhd, solver_mhd,
        boundary_conditions = view_boundary_conditions(mesh_mhd, equations_mhd,
                                                       solver_mhd, coupled_bc))
    semi_euler = SemidiscretizationHyperbolic(
        mesh_euler, equations_euler, initial_condition_euler, solver_euler,
        boundary_conditions = view_boundary_conditions(mesh_euler, equations_euler,
                                                       solver_euler, coupled_bc))
    return SemidiscretizationCoupledP4est(semi_mhd, semi_euler;
                                          coupling_functions = coupling_functions)
end

# Find the output directory from the SaveSolutionCallback in the integrator.
function find_output_dir(integrator)
    for cb in integrator.opts.callback.discrete_callbacks
        hasproperty(cb.affect!, :output_directory) && return cb.affect!.output_directory
    end
    return "out"
end

# ── Combined AMS + AMR callback ───────────────────────────────────────────────

mutable struct AmsAmrCallback
    interval::Int
    tspan_end::Float64
end

function (cb::AmsAmrCallback)(integrator)
    u_ode        = integrator.u
    coupled_semi = integrator.p
    t            = integrator.t
    n_nodes      = nnodes(solver_mhd)

    cell_ids_mhd   = coupled_semi.semis[1].mesh.cell_ids
    cell_ids_euler = coupled_semi.semis[2].mesh.cell_ids

    # 1. Reconstruct the full parent-mesh MHD state (B = 0 in Euler cells).
    u_par_ode = reconstruct_parent_mhd(u_ode, coupled_semi.u_indices,
                                       cell_ids_mhd, cell_ids_euler,
                                       n_nodes, Trixi.ncells(parent_mesh))

    # 2. Build AMR flags from |B|/|B|_max and apply refine!/coarsen!.
    #    A transient parent semidiscretization is created here for its DG cache
    #    (needed by refine!/coarsen! for Jacobian scaling and container reinitialization).
    #    It is local to this callback and GC'd when the function returns — no persistent
    #    parent geometry is stored between callbacks.
    parent_semi = _make_parent_semi()

    alpha  = compute_alpha_mhd(u_par_ode, n_nodes, Trixi.ncells(parent_mesh))
    lambda = make_lambda(alpha, Trixi.current_element_levels(parent_mesh, solver_mhd,
                                                             parent_semi.cache))
    iter_v = Trixi.cfunction(Trixi.copy_to_quad_iter_volume, Val(ndims(parent_mesh)))
    Trixi.iterate_p4est(parent_mesh.p4est, lambda; iter_volume_c = iter_v)

    refined_cells   = Trixi.refine!(parent_mesh)
    Trixi.refine!(u_par_ode, _adaptor, parent_mesh, equations_mhd, solver_mhd,
                  parent_semi.cache, refined_cells)
    coarsened_cells = Trixi.coarsen!(parent_mesh)
    Trixi.coarsen!(u_par_ode, _adaptor, parent_mesh, equations_mhd, solver_mhd,
                   parent_semi.cache, coarsened_cells)

    # 3. Re-split into MHD / Euler views on the updated mesh.
    n_new = Trixi.ncells(parent_mesh)
    new_ids_mhd, new_ids_euler = ams_split_with_buffer(u_par_ode, n_nodes, n_new,
                                                        parent_semi.cache)
    u_new = reshape(u_par_ode, 9, n_nodes, n_nodes, n_new)
    u_mhd_new   = u_new[:, :, :, new_ids_mhd]
    u_euler_new = zeros(4, n_nodes, n_nodes, length(new_ids_euler))
    for (i, k) in enumerate(new_ids_euler)
        u_euler_new[1, :, :, i] .= u_new[1, :, :, k]
        u_euler_new[2, :, :, i] .= u_new[2, :, :, k]
        u_euler_new[3, :, :, i] .= u_new[3, :, :, k]
        for jj in 1:n_nodes, ii in 1:n_nodes
            u_euler_new[4, ii, jj, i] = u_new[5, ii, jj, k] -
                0.5 * (u_new[6,ii,jj,k]^2 + u_new[7,ii,jj,k]^2 + u_new[8,ii,jj,k]^2)
        end
    end

    # 4. Build the new coupled semi, set initial data, and restart.
    semi_new = build_coupled_semi(new_ids_mhd, new_ids_euler)
    ode_new  = semidiscretize(semi_new, (t, cb.tspan_end))
    ode_new.u0[semi_new.u_indices[1]] .= vec(u_mhd_new)
    ode_new.u0[semi_new.u_indices[2]] .= vec(u_euler_new)

    # naccept was restored to _global_naccept[] at each restart, then incremented
    # by new steps, so it already equals the correct cumulative count.
    _global_naccept[] = integrator.stats.naccept

    # Pre-save mesh files with a timestep-stamped name so that initialize_save_cb!
    # (which uses timestep = 0) does not overwrite the files referenced by earlier
    # solution files.
    output_dir = find_output_dir(integrator)
    for i in 1:Trixi.nsystems(semi_new)
        m = semi_new.semis[i].mesh
        m.current_filename = Trixi.save_mesh_file(m, output_dir;
                                                   system = i,
                                                   timestep = _global_naccept[])
        m.unsaved_changes = false
    end
    parent_mesh.unsaved_changes = false  # prevent new views inheriting stale flag

    terminate!(integrator)
    solve(ode_new, CarpenterKennedy2N54(williamson_condition = false);
          dt = 1.0, save_everystep = false,
          ode_default_options()..., callback = integrator.opts.callback)
    return nothing
end

function make_ams_amr_callback(interval, tspan_end)
    cb = AmsAmrCallback(interval, tspan_end)
    condition = (u, t, integrator) -> (integrator.stats.naccept % interval == 0 &&
                                        integrator.stats.naccept > 0)
    return DiscreteCallback(condition, cb, save_positions = (false, false))
end

# ── Initial AMS split ─────────────────────────────────────────────────────────

let
    parent_semi_init = _make_parent_semi()
    u0 = compute_coefficients(0.0, parent_semi_init)
    n0 = Trixi.ncells(parent_mesh)
    global _cell_ids_mhd_init, _cell_ids_euler_init =
        ams_split_with_buffer(u0, nnodes(solver_mhd), n0, parent_semi_init.cache)
    @info "Initial split: $(length(_cell_ids_mhd_init)) MHD cells, " *
          "$(length(_cell_ids_euler_init)) Euler cells"
end  # parent_semi_init is GC'd here

# ── ODE problem and callbacks ─────────────────────────────────────────────────

tspan = (0.0, 20.0)
semi  = build_coupled_semi(_cell_ids_mhd_init, _cell_ids_euler_init)
ode   = semidiscretize(semi, tspan)

summary_callback   = SummaryCallback()
save_solution      = SaveSolutionCallback(interval = 10, save_initial_solution = true,
                                          save_final_solution = true,
                                          solution_variables = cons2prim)
stepsize_callback  = StepsizeCallback(cfl = 0.5)
glm_speed_callback = GlmSpeedCallback(glm_scale = 0.5, cfl = 0.8, semi_indices = [1])
ams_amr_callback   = make_ams_amr_callback(50, tspan[2])

# Restores the step counter at the start of each new solve so that
# SaveSolutionCallback file numbers are continuous across AMS restarts.
counter_fix_cb = DiscreteCallback(
    (u, t, i) -> false, identity;
    save_positions = (false, false),
    initialize = (cb, u, t, integrator) -> (integrator.stats.naccept = _global_naccept[]))

callbacks = CallbackSet(counter_fix_cb, summary_callback, save_solution,
                        stepsize_callback, glm_speed_callback, ams_amr_callback)

sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false);
            dt = 1.0, save_everystep = false,
            ode_default_options()..., callback = callbacks)
summary_callback()
