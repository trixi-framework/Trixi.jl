# CUDA benchmark

This benchmark runs a moderately sized instance of the Taylor-Green-Vortex problem on
NVIDIA GPUs.

Note we currently have to switch to `log_Base` using `LocalPreferences.toml` as otherwise we
wiil see
```
ERROR: LoadError: LLVM error: Undefined external symbol "log"
```
