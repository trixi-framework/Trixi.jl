# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

mutable struct ParaviewCatalystCallback
    interval::Int
end

function Base.show(io::IO,
                   cb::DiscreteCallback{Condition, Affect!}) where {Condition,
                                                                    Affect! <:
                                                                    ParaviewCatalystCallback
                                                                    }
    visualization_callback = cb.affect!
    @unpack interval = visualization_callback
    print(io, "ParaviewCatalystCallback(",
          "interval=", interval,")")
end

function Base.show(io::IO, ::MIME"text/plain",
                   cb::DiscreteCallback{Condition, Affect!}) where {Condition,
                                                                    Affect! <:
                                                                    ParaviewCatalystCallback
                                                                    }
    if get(io, :compact, false)
        show(io, cb)
    else
        visualization_callback = cb.affect!

        setup = [
            "interval" => visualization_callback.interval,
            
        ]
        summary_box(io, "ParaviewCatalystCallback", setup)
    end
end

"""
    ParaviewCatalystCallback(; interval=0,
                            )

Create a callback that visualizes results during a simulation, also known as *in-situ
visualization*.

!!! warning "Experimental implementation"
    This is an experimental feature and may change in any future releases.
"""
function ParaviewCatalystCallback(; interval = 0,
                               )
    mpi_isparallel() && error("this callback does not work in parallel yet")

    ParaviewCatalyst.catalyst_initialize(libpath="/home/nico/Paraview/ParaView-5.13.0-MPI-Linux-Python3.10-x86_64/lib/catalyst")

    visualization_callback = ParaviewCatalystCallback(interval)

    # Warn users if they create a ParaviewCatalystCallback without having loaded the ParaviewCatalyst package
    if !(:ParaviewCatalyst in nameof.(Base.loaded_modules |> values))
        @warn "Package `ParaviewCatalyst` not loaded but required by `ParaviewCatalystCallback` to visualize results"
    end

    DiscreteCallback(visualization_callback, visualization_callback, # the first one is the condition, the second the affect!
                     save_positions = (false, false),
                     initialize = initialize!)
end

function initialize!(cb::DiscreteCallback{Condition, Affect!}, u, t,
                     integrator) where {Condition, Affect! <: ParaviewCatalystCallback}
    visualization_callback = cb.affect!

    visualization_callback(integrator)

    return nothing
end

# this method is called to determine whether the callback should be activated
function (visualization_callback::ParaviewCatalystCallback)(u, t, integrator)
    @unpack interval = visualization_callback

    # With error-based step size control, some steps can be rejected. Thus,
    #   `integrator.iter >= integrator.stats.naccept`
    #    (total #steps)       (#accepted steps)
    # We need to check the number of accepted steps since callbacks are not
    # activated after a rejected step.
    return interval > 0 && (integrator.stats.naccept % interval == 0 ||
            isfinished(integrator))
end

# this method is called when the callback is activated
function (visualization_callback::ParaviewCatalystCallback)(integrator)
    u_ode = integrator.u
    mesh, equations, solver, cache = mesh_equations_solver_cache(integrator.p)
    time = integrator.t
    timestep = integrator.stats.naccept

    leaf_cell_ids = leaf_cells(mesh.tree)                     # Indices der "echten" Zellen
    coordinates = mesh.tree.coordinates[:, leaf_cell_ids]     # Koordinaten der Mittelpunkte der Zellen
    cell_levels = mesh.tree.levels[leaf_cell_ids]                           # Level der Zellen im Baum (pro Level werden die Kanten halbiert)
    mesh.tree.center_level_0                                  # Mittelpunkt auf Level 0, d.h. Mittelpunkt des gesamten Gitters
    length_level_0 = mesh.tree.length_level_0                                  # Kantenlänge auf Level 0
    cell_length = [2.0^-cell_level * length_level_0 for cell_level in cell_levels] 
    min_cell_length = min(cell_length...)

    println()
    println(length(mesh.tree.coordinates[1, leaf_cell_ids]))
    println("*** Catalyst Callback activated")
    println("*** Time ", time)
    println("*** Step ", timestep)
    println("*** u[1] ", u_ode[1])
    println("*** coord[1] ", coordinates[1])
    println()

    # avoid re-evaluating possible FSAL stages
    u_modified!(integrator, false)

    ParaviewCatalyst.ConduitNode() do node
        node["catalyst/state/timestep"] = timestep
        node["catalyst/state/time"] = timestep
        node["catalyst/channels/input/type"] = "mesh"
        node["catalyst/channels/input/data/coordsets/coords/type"] = "uniform"

        pd = nothing
        c_i = 0
        c_j = 0
        c_k = 0
        x0 = 0
        y0 = 0
        z0 = 0
        dx = 0
        dy = 0
        dz = 0
        if ndims(mesh) == 1
            pd = PlotData1D(integrator.u, integrator.p)
            c_i = length(pd.x)
            x0 = min(pd.x...)
            dx = min([pd.x[i + 1] - pd.x[i] for i in 1:(c_i - 1)]...)
        elseif ndims(mesh) == 2
            pd = PlotData2D(integrator.u, integrator.p)
            
            c_i = length(pd.x)
            x0 = min(pd.x...)
            dx = min([pd.x[i + 1] - pd.x[i] for i in 1:(c_i - 1)]...)

            c_j = length(pd.y)
            y0 = min(pd.y...)
            dy = min([pd.y[i + 1] - pd.y[i] for i in 1:(c_j - 1)]...)
        elseif ndims(mesh) == 3
            z_coords = mesh.tree.coordinates[3, leaf_cell_ids]
            z_h1 = [[z_coords[i] - 0.5 * cell_length[i] z_coords[i] + 0.5 * cell_length[i]] for i in 1:length(z_coords)]
            z_h1 = unique!(z_h1)
            z_h = [z_h1[j][k] for j in 1:length(z_h1) for k in 1:2]
            pd_z = [PlotData2D(integrator.u, integrator.p, slice=:xy, point=(0,0,z)) for z in z_h]
            pd = pd_z[1]

            c_i = length(pd.x)
            x0 = min(pd.x...)
            dx = min([pd.x[i + 1] - pd.x[i] for i in 1:(c_i - 1)]...)

            c_j = length(pd.y)
            y0 = min(pd.y...)
            dy = min([pd.y[i + 1] - pd.y[i] for i in 1:(c_j - 1)]...)

            c_k = length(z_h)
            z0 = min(z_h...)
            dz = min([z_h[i + 1] - z_h[i] for i in 1:(c_k - 1)]...)

            #TODO sobald PlotData3D implementiert ersetzen mit 
            # pd = PlotData3D(integrator.u, integrator.p)
            # c_i = length(pd.x)
            # x0 = min(pd.x...)
            # dx = min([pd.x[i + 1] - pd.x[i] for i in 1:(c_i - 1)]...)

            # c_j = length(pd.y)
            # y0 = min(pd.y...)
            # dy = min([pd.y[i + 1] - pd.y[i] for i in 1:(c_j - 1)]...)

            # c_k = length(pd.z)
            # z0 = min(pd.z...)
            # dz = min([pd.z[i + 1] - pd.z[i] for i in 1:(c_k - 1)]...)
        end

        node["catalyst/channels/input/data/coordsets/coords/dims/i"] = c_i
        node["catalyst/channels/input/data/coordsets/coords/origin/x"] = x0
        node["catalyst/channels/input/data/coordsets/coords/spacing/dx"] = dx
        if ndims(mesh) > 1
            node["catalyst/channels/input/data/coordsets/coords/dims/j"] = c_j
            node["catalyst/channels/input/data/coordsets/coords/origin/y"] = y0
            node["catalyst/channels/input/data/coordsets/coords/spacing/dy"] = dy
            if ndims(mesh) > 2
                node["catalyst/channels/input/data/coordsets/coords/dims/k"] = c_k
                node["catalyst/channels/input/data/coordsets/coords/origin/z"] = z0
                node["catalyst/channels/input/data/coordsets/coords/spacing/dz"] = (c_i/c_k) * dx #TODO sobald PlotData3D implementiert auf dz setzen
            end
        end

        node["catalyst/channels/input/data/topologies/mesh/type"] = "uniform"
        node["catalyst/channels/input/data/topologies/mesh/coordset"] = "coords"

        node["catalyst/channels/input/data/fields/solution/association"] = "vertex"
        node["catalyst/channels/input/data/fields/solution/topology"] = "mesh"
        node["catalyst/channels/input/data/fields/solution/volume_dependent"] = "false"
        if ndims(mesh) == 1
            node["catalyst/channels/input/data/fields/solution/values"] = pd.data[1]
        elseif ndims(mesh) == 2
            solution = [pd.data[1][i,j] for j in 1:c_j for i in 1:c_i]
            node["catalyst/channels/input/data/fields/solution/values"] = solution
        elseif ndims(mesh) == 3
            solution_h = [[pd_z[k].data[1][i,j] for j in 1:c_j for i in 1:c_i] for k in 1:c_k]
            solution = [solution_h[i][j] for i in 1:c_k for j in 1:(c_i * c_j)]

            #TODO sobald PlotData3D implementiert, ersetzen durch
            # solution = [pd.data[1][i,j,k] for k in 1:c_k for j in 1:c_j for i in 1:c_i]

            node["catalyst/channels/input/data/fields/solution/values"] = solution
        end
        

        # Conduit.node_info(node) do info_node
        #    Conduit.node_print(info_node, detailed = true)
        # end
        ParaviewCatalyst.catalyst_execute(node)
    end

    return nothing
end

end # @muladd
