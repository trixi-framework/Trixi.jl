# GitHub & Git

This page contains information on how to use GitHub and Git when developing Trixi.jl.

## Development workflow

For adding modifications to Trixi.jl, we generally follow these steps:

### Create an issue (optional)
In many cases it makes sense to start by creating an issue on GitHub. For
example, if the implementation approach for a new feature is not yet clear or if
there should be a discussion about the desired outcome, it is good practice to
first get a consensus on what is the expected result of this modification. A
GitHub issue is *the* place to lead this discussion, as it preserves it in the
project and - together with the actual code changes - allows in the future to revisit
the reasons for a particular choice of implementation or feature.

### Create a branch and *immediately* create a pull request
All feature development, bug fixes etc. should be developed in a branch and not
directly on `main`. If you do not have write access to the main repository on
GitHub, first create a fork of the Trixi.jl repository and clone the fork to
your machine. Then, create a branch locally by executing `git checkout -b
yourbranch`, push it to the repository, and create a pull request (PR).

If you have already cloned Trixi.jl from the main repo to your local machine,
you can also work in that clone. You just need to add your fork as additional
remote repository and push your new branch there.
```bash
git remote add myfork git@github.com:YOUR_NAME/Trixi.jl.git
# get latest main from the main repo
git checkout main
git pull
# create a new branch for a cool new feature, bug fix, ...
git checkout -b YOUR_BRANCH_NAME
# do some work and push it to your fork
git push -u myfork
# go to https://github.com/trixi-framework/Trixi.jl/pull
# and create a PR from your new branch
```

!!! info "Why using pull requests?"
    Immediately creating a PR for your branch has the benefit that all
    code discussions can now be held directly next to the corresponding code. Also,
    the PR allows to easily compare your branch to the upstream branch
    (usually `main`) to see what you have changed. Moreover, tests will run
    automatically.

### Make changes
With a branch and PR in place, you can now write your code and commit
it to your branch. If you request feedback from someone else, make sure to push
your branch to the repository such that the others can easily review your
changes or dive in and change something themselves.

!!! warning "Avoid committing unwanted files"
    When you use `git add .` or similar catch-all versions, make sure you do not
    accidentally commit unwanted files (e.g., Trixi.jl output files, images or
    videos etc.). If it happens anyways, you can undo the last commit (also
    multiple times) by running `git reset HEAD~` (see also [Undo last
    commit](@ref)). However, this strategy only works if you have **not yet
    pushed your changes**. If you *did* push your changes, please talk to one of
    the core developers on how to proceed.

### Keep your branch in sync with `main`
For larger features with longer-living branches, it may make sense to
synchronize your branch with the current `main`, e.g., if there was a bug fix
in `main` that is relevant for you. In this case, perform the following steps to
merge the current `main` to your branch:

  1. Commit all your local changes to your branch and push it. This allows you to
     delete your clone in case you make a mistake and need to abort the merge.
  2. Execute `git fetch` to get the latest changes from the repository.
  3. Make sure you are in the correct branch by checking the output of `git
     status` or by running `git checkout yourbranch`.
  4. Merge main using `git merge main`. If there were no conflicts, hooray!,
     you are done. Otherwise you need to resolve your merge conflicts and commit
     the changes afterwards. A good guide for resolving merge conflicts can be
     found
     [here](https://help.github.com/en/github/collaborating-with-issues-and-pull-requests/resolving-a-merge-conflict-using-the-command-line).

In general, always use `git merge` and not `git rebase` to get the latest
changes from `main`. It is less error-prone and does not create problems on
branches that are worked on collaboratively.

### Prepare for review
If you feel like your branch is ready to be merged to main, prepare it for
review. That is, you should

  * merge the current `main` to your branch
  * run tests if available, but at least ensure that you did not accidentally
    change the results for one of the existing example elixirs
  * properly comment your code
  * delete old/unused code, especially commented lines (unless they contain
    helpful code, in which case you should add a comment on why you keep this
    around)
  * remove debug statements
  * add a `elixir_xxx.jl` that uses your feature (only relevant for new
    features)
  * make sure your code formatting adheres to the [Style guide](@ref)
  * describe changes in `NEWS.md` if appropriate

After you are confident that your branch is cleaned up properly, commit all
changes and push them to the repository.

### Get reviewed
Ask one of the core developers to review your code. Sometimes this will be done
directly, either face-to-face or via a video call. Other times a review will be
conducted asynchronously, with the reviewer leaving comments and annotations. In
some cases it will be necessary to do multiple rounds of reviews, especially if
there are larger changes to be added. Just commit and push your changes to your
branch, and the corresponding pull request will be updated automatically.

Please note that a review has nothing to do with the lack of experience of the
person developing changes: We try to review all code before it gets added to
`main`, even from the most experienced developers. This is good practice and
helps to keep the error rate low while ensuring the the code is developed in a
consistent fashion. Furthermore, do not take criticism of your code personally -
we just try to keep Trixi.jl as accessible and easy to use for everyone.

### Merge branch
Once your branch is reviewed and declared ready for merging by the reviewer,
make sure that all the latest changes have been pushed. Then, one of the
developers will merge your PR. If you are one of the developers, you can also go
to the pull request page on GitHub and and click on **Merge pull request**.
Voilá, you are done! Your branch will have been merged to
`main` and the source branch will have been deleted in the GitHub repository
(if you are not working in your own fork).

### Update your working copy
Once you have merged your branch by accepting the PR on GitHub, you
should clean up your local working copy of the repository by performing the
following steps:

  1. Update your clone by running `git fetch`.
  2. Check out `main` using `git checkout main`.
  3. Delete merged branch locally with `git branch -d yourbranch`.
  4. Remove local references to deleted remote branch by executing `git remote
     prune origin`.

You can now proceed with your next changes by starting again at the top.


## Using Git

### Resources for learning Git
Here are a few resources for learning do use Git that at least one of us found
helpful in the past (roughly ordered from novice to advanced to expert):

  * [Git Handbook by GitHub](https://guides.github.com/introduction/git-handbook/)
  * [Learn Git Branching](https://learngitbranching.js.org/)

### Tips and tricks
This is an unordered collection of different tips and tricks that can be helpful
while working with Git. As usual, your mileage might vary.

#### Undo last commit
If you made a mistake in your last commit, e.g., by committing an unwanted file,
you can undo the latest commit by running
```bash
git reset HEAD~
```
This only works if you have not yet pushed your branch to the GitHub repository.
In this case, please talk to one of the core developers on how to proceed.
Especially when you accidentally committed a large file (image, or video), please
let us know as fast as possible, since the effort to fix the repository grows
considerably over time.

#### Remove large file from repository
If a large file was accidentally committed **and pushed** to the Trixi.jl
repository, please talk to one of the core developers as soon as possible so that they can fix it.

!!! danger "Large files"
    You should never try to fix this yourself, as it potentially
    disrupts/destroys the work of others!

Based on the instructions found
[here](https://rtyley.github.io/bfg-repo-cleaner/) and
[here](https://docs.github.com/en/github/authenticating-to-github/removing-sensitive-data-from-a-repository),
the following steps need to be taken (as documented for GitLab in issue
[#33](https://github.com/trixi-framework/Trixi.jl/issues/33)):

  1. Tell everyone to commit and push their changes to the repository.
  2. Fix the branch in which the file was committed by removing it and committing
     the removal. This is especially important on `main`.
  3. Perform the following steps to clean up the Git repository:
     ```bash
     cd /tmp

     # Download bfg-1.13.0.jar from https://rtyley.github.io/bfg-repo-cleaner/

     # Get fresh clone of repo (so you can throw it away in case there is a problem)
     git clone --mirror git@github.com:trixi-framework/Trixi.jl.git

     # Clean up repo of all files larger than 10M
     java -jar bfg-1.13.0.jar --strip-blobs-bigger-than 10M Trixi.jl.git

     # Enter repo
     cd Trixi.jl.git

     # Clean up reflog and force aggressive garbage collection
     git reflog expire --expire=now --all && git gc --prune=now --aggressive

     # Push changes
     git push

     # Delete clone
     rm -rf Trixi.jl.git
     ```
  4. Tell everyone to clean up their local working copies by performing the
     following steps (also do this yourself):
     ```bash
     # Enter repo
     cd Trixi.jl

     # Get current changes
     git fetch

     # Check out the fixed branch
     git checkout branchname

     # IMPORTANT: Do a rebase instead of a pull!
     git rebase

     # Clean reflog and force garbage collection
     git reflog expire --expire=now --all && git gc --prune=now --aggressive
     ```
     **IMPORTANT**: You need to do a `git rebase` instead of a `git pull` when
     updating the fixed branch.
