# This file is just for testing which packages are installed in the notebook (via binder).
#nb # Just a solution to make this work might be:
#nb using Pkg
#nb Pkg.activate("../../../")
#nb Pkg.instantiate()

# In the root Project.toml the package Printf is installed
using Printf
@printf "%.4f" 0.123456789

# In docs/Project.toml Plots is installed.
using Plots
plot(1:5, rand(5))