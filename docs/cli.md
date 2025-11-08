# Command-Line Interface (CLI)

## 1. Install

Install `max-div` as a local tool, e.g:

```bash
uv tool install max-div[numba]
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
--turbo       Run a much faster (but less reliable) benchmark
--markdown    Output results in Markdown format
```