from max_div._core._cli.bm_speed import SpeedParam

# Every internal benchmark times its function through `benchmark()` with the same three knobs;
# the speed-dependent values are shared here, module-specific parameters stay in their modules.
TIME_PER_RUN_SEC = SpeedParam(0.01, 1e-5)
N_WARMUP = SpeedParam(8, 2, scale="linear")
N_BENCHMARK = SpeedParam(25, 1, scale="linear")
