using LinearAlgebra
using Convex
using ECOS
const MOI = Convex.MOI

function filter_Eigvals(EigVals::Array{Complex{Float64}}, threshold:: Float64)
    filtered_eigvals_counter = 0
    #FilteredEigVals is not an Array but the code seems to work fine regardless
    FilteredEigVals = Complex{Float64}[]
    for EigVal in EigVals
        if abs(EigVal) < threshold
            filtered_eigvals_counter += 1
        else
            push!(FilteredEigVals, EigVal)
        end
    end

    println("Total number of $filtered_eigvals_counter eigenvalues has not been recorded because they were smaller than $threshold")

    return length(FilteredEigVals), FilteredEigVals
end

function ReadInEigVals(Path_toEvalFile::AbstractString)

  NumEigVals = -1 # Declare and set to some value
  open(Path_toEvalFile, "r") do EvalFile
    NumEigVals = countlines(EvalFile)
  end

  EigVals = Array{Complex{Float64}}(undef, NumEigVals)

  LineIndex = 0
  open(Path_toEvalFile, "r") do EvalFile
    # read till end of file
    while !eof(EvalFile) 
    
      # read a new / next line for every iteration          
      LineContent = readline(EvalFile)     
  
      EigVals[LineIndex + 1] = parse(Complex{Float64}, LineContent)

      LineIndex += 1
    end
  end

  return NumEigVals, EigVals
end

function Polynoms(ConsOrder::Int, NumStages::Int, NumEigVals::Int, normalized_powered_EigvalsScaled::Array{Complex{Float64}}, pnoms::Array{Complex{Float64}}, gamma::Variable)

  for i in 1:NumEigVals
    pnoms[i] = 1.0
  end

  for k in 1:ConsOrder
    for i in 1:NumEigVals
      pnoms[i] += normalized_powered_EigvalsScaled[i,k]
    end
  end

  for k in ConsOrder + 1:NumStages
    pnoms += gamma[k - ConsOrder] * normalized_powered_EigvalsScaled[:,k]
  end
  
  return maximum(abs(pnoms))
end


function Bisection(ConsOrder::Int, NumEigVals::Int, NumStages::Int, dtMax::Float64, dtEps::Float64, EigVals::Array{Complex{Float64}})

  dtMin = 0.0

  dt    = -1.0

  AbsP  = -1.0

  pnoms = ones(Complex{Float64}, NumEigVals, 1)
  
  gamma = Variable(NumStages - ConsOrder) # Init datastructure for results

  normalized_powered_Eigvals = zeros(Complex{Float64}, NumEigVals, NumStages)

  for j in 1:NumStages
    fac_j = factorial(j)
    for i in 1:NumEigVals
      normalized_powered_Eigvals[i, j] =  EigVals[i]^j / fac_j
    end
  end

  normalized_powered_EigvalsScaled = zeros(Complex{Float64}, NumEigVals, NumStages)

  while dtMax - dtMin > dtEps
    dt = 0.5 * (dtMax + dtMin)

    for k in 1:NumStages
      dt_k = dt^k
      for i in 1:NumEigVals
        normalized_powered_EigvalsScaled[i,k] = dt_k * normalized_powered_Eigvals[i,k]
      end
    end

    # Use last optimal values for gamm0 in (potentially) next iteration
    problem = minimize(Polynoms(ConsOrder, NumStages, NumEigVals, normalized_powered_EigvalsScaled, pnoms, gamma))

    Convex.solve!(
      problem,
      # Parameters taken from default values for EiCOS
      MOI.OptimizerWithAttributes(ECOS.Optimizer, "gamma" => 0.99,
                                                   "delta" => 2e-7,
                                                   "eps" => 1e-13, # 1e-13
                                                   "feastol" => 1e-9, # 1e-9
                                                   "abstol" => 1e-9, # 1e-9
                                                   "reltol" => 1e-9, # 1e-9
                                                   "feastol_inacc" => 1e-4,
                                                   "abstol_inacc" => 5e-5,
                                                   "reltol_inacc" => 5e-5,
                                                   "nitref" => 9,
                                                   "maxit" => 100,
                                                   "verbose" => 3); silent_solver = false
    )

    AbsP = problem.optval

    println("Current MaxAbsP: ", AbsP, "\nCurrent dt: ", dt, "\n")

    if AbsP < 1.0
      dtMin = dt
    else
      dtMax = dt
    end
  end

  return evaluate(gamma), AbsP, dt
end

function undo_normalization(ConsOrder::Int, NumStages::Int, gammaOpt)
  for k in ConsOrder + 1:NumStages
    gammaOpt[k - ConsOrder] = gammaOpt[k - ConsOrder] / factorial(k)
  end
  return gammaOpt
end