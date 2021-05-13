# This file is just for testing which packages are installed in the notebook (via binder).
# In Trixi Project.toml there is Printf
using Printf
@printf "%.4f" 0.123456789

# In docs/Project.toml Plots is installed.
using Plots
plot(1:5, rand(5))