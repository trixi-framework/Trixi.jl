# Tutorial 3: Testing
[![](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/bennibolm/Trixi.jl/tutorials?filepath=docs/src/tutorials/notebooks/testing_repository.ipynb)
[![](https://img.shields.io/badge/show-nbviewer-579ACA.svg)](https://nbviewer.jupyter.org/github/bennibolm/Trixi.jl/tree/tutorials/docs/src/tutorials/notebooks/testing_repository.ipynb)

This file is just for testing which packages are installed in the notebook (via binder).

In the root Project.toml the package Printf is installed

```julia
using Printf
@printf "%.4f" 0.123456789
```

```
0.1235
```

In docs/Project.toml Plots is installed.

```julia
using Plots
plot(1:5, rand(5))
```
![](423885516.png)

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

