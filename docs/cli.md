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

There are various (currently 1, to be extended) benchmarking commands available with similar behavior:

```bash
max-div benchmark sample-int
```

All benchmarking commands support the following shared flags & options:
```
--turbo       Run a much faster (but less reliable) benchmark; equivalent with --speed=1.0; overrides --speed if both are provided.
--speed       Value between 0.0 (default) and 1.0 (fastest) to speed up the benchmark at the cost of accuracy.
--markdown    Output results in Markdown format
```