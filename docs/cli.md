# Command-Line Interface (CLI)

## 1. Install

Install `max-div` as a local tool, e.g:

```bash
uv tool install max-div
```

This will then install the system-wide command `max-div`.

## 2. Usage

### 2.1. Help

To see the available commands and options, run:

```bash
max-div --help
```

### 2.2. Benchmarking

There are two categories of benchmarks: `solver` and `internal`.

```bash
max-div benchmark solver <test_problem>
max-div benchmark internal <method>
```

#### 2.2.1. Solver benchmarks

The following commands support running test problems to evaluate solver performance:

```bash
max-div benchmark solver list
max-div benchmark solver run <test_problem>
```

The `run` sub-command supports the following flags & options:

```
--markdown    Output results in Markdown format
```

For results, see [here](./benchmarks/solver/_index.md).

#### 2.2.2. Internal benchmarks

There are various benchmarking commands available for internal functionality:

```bash
max-div benchmark internal randint
max-div benchmark internal randint_constrained
max-div benchmark internal diversity_metrics
max-div benchmark internal modify_p_selectivity
```

All benchmarking commands support the following shared flags & options:
```
--turbo       Run a much faster (but less reliable) benchmark; equivalent with --speed=1.0; overrides --speed if both are provided.
--speed       Value between 0.0 (default) and 1.0 (fastest) to speed up the benchmark at the cost of accuracy.
--markdown    Output results in Markdown format
```

For results, see [here](./benchmarks/internal/_index.md).

### 2.3. Numba status

```bash
max-div numba-status
```

Results in the following:
```
Numba version    : 0.62.1
llvmlite version : 0.45.1

Numba Configuration:
--------------------------------------------------
SVML enabled       : False
Threading layer    : default
Number of threads  : 16
Debug mode         : 0
Optimization level : _OptLevel(3)
Disable JIT        : 0
--------------------------------------------------
```