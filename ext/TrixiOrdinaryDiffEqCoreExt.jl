module TrixiOrdinaryDiffEqCoreExt

import Trixi: load_controller!, store_controller!
import OrdinaryDiffEqCore: OrdinaryDiffEqCore, PIController, PIDController
import HDF5: attributes

@static if Base.pkgversion(OrdinaryDiffEqCore) >= v"4"
    import OrdinaryDiffEqCore: PIControllerCache, PIDControllerCache
end

# Support to load controller.
# OrdinaryDiffEqCore < v4: The state of PIController and PIDController lives on the
# integrator (integrator.qold) rather than on the controller struct.
# OrdinaryDiffEqCore >= v4: the legacy controllers are replaced by PIControllerCache and
# PIDControllerCache, and integrator.qold no longer exists.

# OrdinaryDiffEqCore < v4, PI controller:
# Previous error estimate stored as integrator.qold.
function load_controller!(integrator, ::PIController, file)
    if !("time_integrator_qold" in keys(attributes(file)))
        error("Missing data in restart file: check the consistency of adaptive time controller with initial setup!")
    end
    integrator.qold = read(attributes(file)["time_integrator_qold"])
end

# OrdinaryDiffEqCore < v4, PID controller:
# Accumulated dt factor stored as integrator.qold.
# Error history field layout changed between v3.31 and v3.32:
#   <= v3.31: err::MVector{3} (single array field)
#   v3.32–v3.33: err1, err2, err3 (individual scalar fields)
function load_controller!(integrator, controller::PIDController, file)
    if !("time_integrator_qold" in keys(attributes(file)) ||
         !("time_integrator_controller_err" in keys(attributes(file))))
        error("Missing data in restart file: check the consistency of adaptive time controller with initial setup!")
    end
    integrator.qold = read(attributes(file)["time_integrator_qold"])
    if hasproperty(controller, :err)
        # OrdinaryDiffEqCore <= v3.31
        controller.err[:] = read(attributes(file)["time_integrator_controller_err"])
    elseif hasproperty(controller, :err1) && hasproperty(controller, :err2) &&
           hasproperty(controller, :err3)
        # OrdinaryDiffEqCore v3.32–v3.33
        err = read(attributes(file)["time_integrator_controller_err"])
        controller.err1 = err[1]
        controller.err2 = err[2]
        controller.err3 = err[3]
    end
end

@static if Base.pkgversion(OrdinaryDiffEqCore) >= v"4"
    # OrdinaryDiffEqCore >= v4, PI controller:
    # Previous error estimate stored as PIControllerCache.errold (integrator.qold removed).
    function load_controller!(integrator, controller::PIControllerCache, file)
        if !("time_integrator_qold" in keys(attributes(file)))
            error("Missing data in restart file: check the consistency of adaptive time controller with initial setup!")
        end
        controller.errold = read(attributes(file)["time_integrator_qold"])
    end

    # OrdinaryDiffEqCore >= v4, PID controller:
    # Accumulated dt factor stored as PIDControllerCache.dt_factor (integrator.qold removed).
    # Error history stored as PIDControllerCache.err::Vector{3}.
    function load_controller!(integrator, controller::PIDControllerCache, file)
        if !("time_integrator_qold" in keys(attributes(file)) ||
             !("time_integrator_controller_err" in keys(attributes(file))))
            error("Missing data in restart file: check the consistency of adaptive time controller with initial setup!")
        end
        controller.dt_factor = read(attributes(file)["time_integrator_qold"])
        controller.err[:] = read(attributes(file)["time_integrator_controller_err"])
    end
end

# OrdinaryDiffEqCore < v4, PI controller:
# Previous error estimate stored as integrator.qold.
function store_controller!(file, ::PIController, integrator)
    attributes(file)["time_integrator_qold"] = integrator.qold
end

# OrdinaryDiffEqCore < v4, PID controller:
# Accumulated dt factor stored as integrator.qold.
# Error history field layout changed between v3.31 and v3.32:
#   <= v3.31: err::MVector{3} (single array field)
#   v3.32–v3.33: err1, err2, err3 (individual scalar fields)
function store_controller!(file, controller::PIDController, integrator)
    attributes(file)["time_integrator_qold"] = integrator.qold
    if hasproperty(controller, :err)
        # OrdinaryDiffEqCore <= v3.31
        attributes(file)["time_integrator_controller_err"] = controller.err
    else
        # OrdinaryDiffEqCore v3.32–v3.33
        attributes(file)["time_integrator_controller_err"] = [controller.err1,
            controller.err2,
            controller.err3]
    end
end

@static if Base.pkgversion(OrdinaryDiffEqCore) >= v"4"
    # OrdinaryDiffEqCore >= v4, PI controller:
    # Previous error estimate stored as PIControllerCache.errold (integrator.qold removed).
    function store_controller!(file, controller::PIControllerCache, integrator)
        attributes(file)["time_integrator_qold"] = controller.errold
    end

    # OrdinaryDiffEqCore >= v4, PID controller:
    # Accumulated dt factor stored as PIDControllerCache.dt_factor (integrator.qold removed).
    # Error history stored as PIDControllerCache.err::Vector{3}.
    function store_controller!(file, controller::PIDControllerCache, integrator)
        attributes(file)["time_integrator_qold"] = controller.dt_factor
        attributes(file)["time_integrator_controller_err"] = controller.err
    end
end

end
