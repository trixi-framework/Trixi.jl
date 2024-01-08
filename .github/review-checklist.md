### Review checklist

This checklist is meant to assist creators of PRs (to let them know what reviewers will typically look for) and reviewers (to guide them in a structured review process). Items do not need to be checked explicitly for a PR to be eligible for merging.

#### Purpose and scope
- [ ] The PR has a single goal that is clear from the PR title and/or description.
- [ ] All code changes represent a single set of modifications that logically belong together.
- [ ] No more than 500 lines of code are changed or there is no obvious way to split the PR into multiple PRs.

#### Code quality
- [ ] The code can be understood easily.
- [ ] Newly introduced names for variables etc. are self-descriptive and consistent with existing naming conventions.
- [ ] There are no redundancies that can be removed by simple modularization/refactoring.
- [ ] There are no leftover debug statements or commented code sections.
- [ ] The code adheres to our [conventions](https://trixi-framework.github.io/Trixi.jl/stable/conventions/) and [style guide](https://trixi-framework.github.io/Trixi.jl/stable/styleguide/), and to the [Julia guidelines](https://docs.julialang.org/en/v1/manual/style-guide/).

#### Documentation
- [ ] New functions and types are documented with a docstring or top-level comment.
- [ ] Relevant publications are referenced in docstrings (see [example](https://github.com/trixi-framework/Trixi.jl/blob/7f83a1a938eecd9b841efe215a6e482e67cfdcc1/src/equations/compressible_euler_2d.jl#L601-L615) for formatting).
- [ ] Inline comments are used to document longer or unusual code sections.
- [ ] Comments describe intent ("why?") and not just functionality ("what?").
- [ ] If the PR introduces a significant change or new feature, it is documented in `NEWS.md`.

#### Testing
- [ ] The PR passes all tests.
- [ ] New or modified lines of code are covered by tests.
- [ ] New or modified tests run in less then 10 seconds.

#### Performance
- [ ] There are no type instabilities or memory allocations in performance-critical parts.
- [ ] If the PR intent is to improve performance, before/after [time measurements](https://trixi-framework.github.io/Trixi.jl/stable/performance/#Manual-benchmarking) are posted in the PR.

#### Verification
- [ ] The correctness of the code was verified using appropriate tests.
- [ ] If new equations/methods are added, a convergence test has been run and the results
  are posted in the PR.

*Created with :heart: by the Trixi.jl community.*