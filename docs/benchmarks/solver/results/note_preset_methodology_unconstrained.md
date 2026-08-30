Some notes:

- **Each benchmark run gets a different seed**, so the scatter between neighboring budget points also gives an idea of result variability.
- **Budgets are end-to-end**: they include the initial setup cost (distance computations, worker setup).
- The q10-q50-q90 trend lines are obtained using **cubic-spline quantile regression** through that scatter.
- The runs compare the four presets **using 1 worker**, plus the default preset **under default multi-worker settings** (for the benchmark machine used).
