from max_div._core._cli.benchmarks._helpers.speed_scaling import SpeedParam

# Every internal benchmark times its function through `benchmark()` with the same settings;
# the speed-dependent values are shared here, module-specific parameters stay in their modules.
TIME_PER_RUN_SEC = SpeedParam(slow=0.01, fast=1e-5)
N_WARMUP = SpeedParam(slow=8, fast=2, scale="linear")
N_BENCHMARK = SpeedParam(slow=25, fast=1, scale="linear")
