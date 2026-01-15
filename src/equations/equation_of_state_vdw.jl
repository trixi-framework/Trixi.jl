# assume mass basis for simplicity 
struct VanDerWaals{RealT} <: AbstractEquationOfState
    a::RealT
    b::RealT
    R::RealT
    gamma::RealT
    cv::RealT
end

# by default, van der Waals parameters are for N2
function VanDerWaals(; a=174.64049524257663, b=0.001381308696129041, 
                 gamma = 5/3, R=296.8390795484912)
    cv = R / (gamma - 1)
    return VanDerWaals(promote(a, b, R, gamma, cv)...)
end

function pressure(V, T, eos::VanDerWaals)
    (; a, b, R) = eos
    rho = inv(V)
    p = rho * R * T / (1 - rho * b) - a * rho^2
    return p
end

function internal_energy(V, T, eos::VanDerWaals) 
    (; cv, a) = eos
    rho = inv(V)
    e = cv * T - a * rho 
    return e 
end

function specific_entropy(V, T, eos::VanDerWaals)
    (; cv, b, R) = eos
    s = cv * log(T) + R * log(V - b) # + s0 = -319.1595051898981 for consistency with Clapeyron.jl
    return s
end

function speed_of_sound(V, T, eos::VanDerWaals)
    (; a, b, gamma) = eos
    rho = inv(V)
    e = internal_energy(V, T, eos)
    c2 = gamma * (gamma - 1) * (e + rho * a) / (1 - rho * b)^2 - 2 * a * rho
    return sqrt(c2)
end

