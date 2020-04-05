# Trixi: A tree-based flexible DG/SBP framework written in Julia

<p align="center">
  <img width="300px"
       src="https://gitlab.mi.uni-koeln.de/numsim/code/Trixi.jl/-/raw/master/docs/src/assets/logo.png">
</p>

*Trixi* is a flexible DG/SBP framework written in the Julia programming
language. It is based on a two-dimensional hierarchical mesh (quadtree) and aims
to be easy to use and extend.


## Installation
Install Trixi, the postprocessing tools, and the respective dependencies by
performing the following steps:

1.  Clone the repository:
    ```bash
    git clone git@gitlab.mi.uni-koeln.de:numsim/code/Trixi.jl.git
    ```
2.  Enter the cloned directory and run the following command to install all
    required dependencies:
    ```bash
    julia utils/install.jl
    ```


## Usage
Enter the root directory `Trixi.jl/` and run
```bash
bin/trixi parameters.toml
```

To change the simulation setup, edit `parameters.toml`. You can also pass a different
parameters file on the command line, e.g., `bin/trixi awesome_parameters.toml`.


## Documentation
If you are on GitLab, you can browse the documentation
[online](https://numsim.gitlab-pages.sloede.com/personal/mschlott/Trixi.jl/). If
you are in your terminal, you will find all documentation under
[`docs/src`](docs/src).


## Authors
Trixi was created by Michael Schlottke-Lakemper and Gregor Gassner.
