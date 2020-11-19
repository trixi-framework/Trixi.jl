#using Distributions: Uniform

function trainfunction1d(func,x)
    if func == 1
        u = sin(4*pi*x) 
    elseif func == 2
        u = sin(2*pi*x) 
    elseif func == 3
        u = sin(pi*x) 
    elseif func == 4
        u = sin(0.1*pi*x) 
    elseif func == 5
        u = sin(0.5*pi*x) 
    elseif func == 6
        u = sin(6*pi*x) 
    elseif func == 7
        u = sin(3*pi*x)+sin(2*pi*x)*cos(5*pi*x)
    elseif func == 8
        u = sin(1*pi*x)+sin(4*pi*x)*cos(2*pi*x)
    elseif func == 9
        u = sin(4*pi*x)+sin(0.2*pi*x)*cos(6*pi*x)
    elseif func == 10
        u = sin(pi*x)+sin(3*pi*x)*cos(0.5*pi*x)
    elseif func == 11
        u =  2*x
    elseif func == 12
        u =  6*x
    elseif func == 13
        u =  -4*x
    elseif func == 14
        u =  20*x
    elseif func == 15
        u =  -40*x
    elseif func == 16
        u =  0.2*x
    elseif func == 17
        u =  -0.6*x
    elseif func == 18
        u =  -2*x
    elseif func == 19
        u = 5 * abs(x)
    elseif func == 20
        u = -2 * abs(x)
    elseif func == 21
        u = 15 * abs(x)
    elseif func == 22
        u = -0.3 * abs(x)
    elseif func == 23
        u = -50 * abs(x)
    elseif func == 24
        u = 0.8 * abs(x)
    elseif func == 25
        u = 0.1 * abs(x)
    elseif func == 26
        u = -4 * abs(x)
    end
    return u
end

function Output(func, xi, h)
    if func <= 18
      Y = [0; 1]
    elseif func > 18 && func <=26
      if 0 >= xi-(3/2)*h && 0 <= xi+(3/2)*h 
        Y = [1; 0]
      else
        Y = [0; 1]
      end
    elseif func == 27
      Y = [1; 0]
    end
    return Y
end

function troubledcellfunctionstep(x, ul, ur, x0)
    u = ul*(x<x0) + ur*(x>x0)
    return u
end

function good_cell(node_coord, h, x0)
    x = node_coord[1] + h/2
    if x0 < x+(3/2)*h && x0 > x-(3/2)*h 
        return false
    else 
        return true
    end
        
end


function validfunction1d(func,x)
    if func == 1
        u = sin(2*pi*x) 
    elseif func == 2
        u = sin(1*pi*x)
    elseif func == 3
        u = sin(1*pi*x) + sin(2*pi*x)
    elseif func == 4
        u = sin(1*pi*x) +sin(2*pi*x)+ sin(3*pi*x) 
    elseif func == 5
        u = sin(1*pi*x)+sin(2*pi*x)+sin(2*pi*x)+sin(4*pi*x)+sin(5*pi*x)
    elseif func == 6
        u = sin(4*pi*x) 
    elseif func == 7
        u = sin(3*pi*x)*sin(2*pi*x)*cos(5*pi*x)
    elseif func == 8
        u = sin(1*pi*x)*sin(4*pi*x)*cos(2*pi*x)
    elseif func == 9
        u = sin(4*pi*x)*sin(0.2*pi*x)*cos(6*pi*x)
    elseif func == 10
        u = sin(pi*x)*sin(3*pi*x)*cos(0.5*pi*x)
    elseif func == 11
        u =  sin(0.2pi*x)*sin(2*pi*x)*cos(3.5*pi*x)
    elseif func == 12
        u =  sin(pi*x) + exp(x)
    elseif func == 13
        u =  sin(2*pi*x) + exp(x)
    elseif func == 14
        u =  sin(0.2*pi*x) + exp(x)
    elseif func == 15
        u =  sin(3.2*pi*x) + exp(x)
    elseif func == 16
        u = sin(5*pi*x) + exp(x)
    end
    return u
end





