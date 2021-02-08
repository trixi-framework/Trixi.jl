# Callbacks

Many of the advanced features of Trixi such as adaptive mesh refinement are implemented as 
callbacks. A callback is an algorithmic entity that gets passed to the ODE solver and 
is called at specific points during execution to perform certain tasks. Callbacks in Trixi are 
either called after each time step or after each stage of the ODE 
solver. 

## Overview

### CFL-based time step control
Time step control can be performed with a [`StepsizeCallback`](@ref). 

### Adaptive Mesh Refinement
Trixi uses a hierarchical Cartesian mesh which can be locally refined in a solution-adaptive way. 
This can be used to speed up simulations with minimal loss in overall accuracy. AMR can be used by 
passing an [`AMRCallback`](@ref) to the ODE solver. The `AMRCallback` requires a controller such as 
the [`ControllerThreeLevel`](@ref) or [`ControllerThreeLevelCombined`](@ref) to tell the AMR 
algorithm which cells to refine/coarsen. 

### Analyzing the numerical solution
The [`AnalysisCallback`](@ref) can be used to analyze the numerical solution, e.g. calculate 
errors or user-specified integrals, and print the results to the screen. The results can also be 
saved in a file.

### I/O
To save the solution in regular intervals you can use a [`SaveSolutionCallback`](@ref). It is also 
possible to create restart files using the [`SaveRestartCallback`](@ref). 

### Misc
* The [`AliveCallback`](@ref) prints some information to the screen to show that a simulation is 
  still running.
* [`SummaryCallback`](@ref): Prints a human-readable summary of the simulation setup and controls 
  the timer.
* [`VisualizationCallback`](@ref): Used for in-situ visualization. See 
  [Visualizing results during a simulation](@ref).

## Usage
Callbacks that are called after each time step are passed to the `solve` method from the ODE solver 
via the keyword argument `callback`.
If you want to use a single callback `cb`, it is sufficient to pass it as `callback=cb`. 
When using two or more callbacks, you need to turn them into a `CallbackSet` first by calling 
`callbacks = CallbackSet(cb1, cb2)` and passing it as `callback=callbacks`.

!!! note
    There are some restrictions regarding the order of callbacks in a `CallbackSet`.
   
    The callbacks are called *after* each time step but some callbacks actually belong to the next 
    time step. Therefore, the callbacks should be ordered in the following way:
    * Callbacks that belong to the current time step:
        * `SummaryCallback` controls, among other things, timers and should thus be first
        * `AnalysisCallback` may do some checks that mark a time step as the last one
        * `AliveCallback` should be nearby `AnalysisCallback`
        * `SaveSolutionCallback`/`SaveRestartCallback` should save the current solution before it is 
          degraded by AMR
        * `VisualizationCallback` should be called before the mesh is adapted
    * Callbacks that belong to the next time step:
        * `AMRCallback`
        * `StepsizeCallback` must be called after `AMRCallback` to accomodate potential changes to 
          the mesh
