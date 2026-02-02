module TrixiOrdinaryDiffEqCore

import Trixi: load_controller!
import OrdinaryDiffEqCore: OrdinaryDiffEqCore, PIController, PIDController

@static if Base.pkgversion(OrdinaryDiffEqCore) >= v"3.4"
    import OrdinaryDiffEqCore: PIControllerCache, PIDControllerCache
end

# Support to load controller
function load_controller!(integrator, controller::PIController, file)
    if !("time_integrator_qold" in keys(attributes(file)))
        error("Missing data in restart file: check the consistency of adaptive time controller with initial setup!")
    end
    integrator.qold = read(attributes(file)["time_integrator_qold"])
end

function load_controller!(integrator, controller::PIDController, file)
    if !("time_integrator_qold" in keys(attributes(file)) ||
         !("time_integrator_controller_err" in keys(attributes(file))))
        error("Missing data in restart file: check the consistency of adaptive time controller with initial setup!")
    end
    integrator.qold = read(attributes(file)["time_integrator_qold"])
    controller.err[:] = read(attributes(file)["time_integrator_controller_err"])
end

@static if Base.pkgversion(OrdinaryDiffEqCore) >= v"3.4"
    function load_controller!(integrator, controller::PIControllerCache, file)
        if !("time_integrator_qold" in keys(attributes(file)))
            error("Missing data in restart file: check the consistency of adaptive time controller with initial setup!")
        end
        controller.errold = read(attributes(file)["time_integrator_controller_err"])
        integrator.qold = read(attributes(file)["time_integrator_qold"])
    end

    function load_controller!(integrator, controller::PIDControllerCache, file)
        if !("time_integrator_qold" in keys(attributes(file)) ||
             !("time_integrator_controller_err" in keys(attributes(file))))
            error("Missing data in restart file: check the consistency of adaptive time controller with initial setup!")
        end
        controller.dt_factor = integrator.qold = read(attributes(file)["time_integrator_qold"])
        controller.err[:] = read(attributes(file)["time_integrator_controller_err"])
    end
end

end
