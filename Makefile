file_path=

# --- how the package suite is installed and run ------------
# Single definition, shared by every target below AND by the CI test matrix, which calls these
# targets rather than repeating the command. That is the point: the interpreter, the resolution
# strategy and the dependency group cannot drift between a developer machine and CI, because
# there is only one place that names them. CI overrides PY and RESOLUTION per matrix leg; the
# defaults are the single representative combo to run locally.
PY ?= 3.13
RESOLUTION ?= highest
UV_RUN = uv run --python $(PY) --resolution $(RESOLUTION) --group dev

# Appended to the pytest invocation. Locally this silences the warning summary; CI overrides it
# to pass coverage flags, and deliberately keeps the warnings visible.
PYTEST_ARGS ?= --disable-warnings

.PHONY: help build test collect-test-ids coverage test-and-coverage format format-single-file lint dev-setup release splash docs show-coverage show-docs update-internal-benchmarks update-solver-strategies-benchmarks update-all-benchmarks update-solver-benchmark-figures

help:
	@echo 'Commands:'
	@echo ''
	@echo '  help                                   Show this help message.'
	@echo ''
	@echo '  build                                  (Re)build package using uv.'
	@echo ''
	@echo '  test                                   Run pytest unit tests.'
	@echo '  collect-test-ids                       List collected test node-ids without running them.'
	@echo '  coverage                               Generate test coverage report. (./reports/coverage)'
	@echo '  test-and-coverage                      Run unit tests (=with numba) + generate coverage report (=without numba).'
	@echo ''
	@echo '  format                                 Format source code using ruff.'
	@echo '  format-single-file                     Format single file using ruff. Useful in e.g. PyCharm to automatically trigger formatting on file save.'
	@echo '  lint                                   Run all pre-commit hooks on all files.'
	@echo '  dev-setup                              Sync dev environment & install pre-commit hooks.'
	@echo ''
	@echo '  release                                Release a new version. Usage: make release VERSION=X.Y.Z'
	@echo ''
	@echo '  splash                                 Build splash screen using current version of package. (./images/splash_with_version.webp)'
	@echo '  docs                                   Update mkdocs site. (./reports/docs)'
	@echo ''
	@echo '  show-coverage                          Open test coverage report in browser.'
	@echo '  show-docs                              Open mkdocs site in browser.'
	@echo ''
	@echo '  update-internal-benchmarks             Run internal benchmarks and update results in ./docs/benchmarks/internal/results/.'
	@echo '  update-solver-strategies-benchmarks    Run solver benchmarks and update results in ./docs/benchmarks/solver/results/.'
	@echo '  update-all-benchmarks                  Run all benchmarks and update results in ./docs/benchmarks/.'
	@echo '  update-solver-benchmark-figures        Updates all benchmark-related figures in ./docs/benchmarks/solver/images.'
	@echo ''
	@echo 'Options:'
	@echo ''
	@echo '  format-single-file             - accepts `file_path=<path>` to pass the relative path of the file to be formatted.'
	@echo '  test / collect-test-ids / coverage'
	@echo '                                 - accept `PY=<version>` and `RESOLUTION=highest|lowest-direct` to pick the'
	@echo '                                   interpreter and uv resolution strategy, and `PYTEST_ARGS=<flags>` to replace'
	@echo '                                   the default pytest flags. The CI matrix drives these targets that way.'

build:
	uv build;

test:
	# run all tests - with numba & one interpreter (see PY / RESOLUTION above)
	$(UV_RUN) pytest ./tests --durations=20 $(PYTEST_ARGS)

# Collected node-ids, one per line. CI unions these across matrix legs to count the suite, so this
# target's stdout is data: it is written with `@` and its commentary lives here rather than in the
# recipe, since an echoed recipe line would land in whatever consumes the list.
# -o addopts="" clears `-n auto`, so collection runs in-process instead of under xdist.
collect-test-ids:
	@$(UV_RUN) pytest ./tests --collect-only -q -o addopts=""

test-benchmarks:
	# comparison-benchmark harness tests - separate from the package suite (needs the benchmarks deps groups)
	uv run --group benchmarks --group benchmarks-gpl --python 3.13 pytest ./benchmarks/tests --durations=20 --disable-warnings

coverage:
	# NOTE: NUMBA_DISABLE_JIT ensure coverage collects detailed line-by-line coverage info, also for numba-compiled functions
    #       NUMBA_JIT_COVERAGE is another option, but would incorrectly emit coverage info for ALL compiled lines, when a function is triggered.
	mkdir -p ./reports
	# same install surface as `test` and as the CI coverage legs (see PY / RESOLUTION above)
	NUMBA_DISABLE_JIT=1 COVERAGE_FILE=./reports/.coverage $(UV_RUN) pytest ./tests --cov --cov-report=html:./reports/coverage --durations=20 $(PYTEST_ARGS)

test-and-coverage:
	$(MAKE) test;
	$(MAKE) coverage;

format:
	uv run ruff format .;
	uv run ruff check --fix .;

format-single-file:
	uv run ruff format ${file_path};
	uv run ruff check --fix ${file_path};

lint:
	uv run pre-commit run --all-files

dev-setup:
	uv sync --all-extras
	uv run pre-commit install

release:
ifndef VERSION
	$(error VERSION is required: make release VERSION=X.Y.Z)
endif
	$(MAKE) test
	uv run python scripts/release.py $(VERSION)

splash:
	./images/splash/create_splash_with_version.sh "$$(uv version --short)-dev";

docs:
	# output dir comes from mkdocs.yml (site_dir: ./reports/docs)
	uv run --extra docs mkdocs build;

show-coverage:
	# trigger browser to open coverage report
	open ./reports/coverage/index.html

show-docs:
	# trigger browser to open mkdocs site
	# (go through pycharm built-in server, to make this site work correctly)
	open http://localhost:63342/max-div/reports/docs/index.html

update-internal-benchmarks:
	uv run max-div benchmark internal --markdown --file all;
	rm ./docs/benchmarks/internal/results/*.md;
	mv ./benchmark*.md ./docs/benchmarks/internal/results/;
	$(MAKE) docs

update-solver-strategies-benchmarks:
	uv run max-div benchmark solver strategies --markdown --file --problem=all;
	mv -f ./benchmark*.md ./docs/benchmarks/solver/results/;
	$(MAKE) docs

update-all-benchmarks: update-internal-benchmarks update-solver-strategies-benchmarks

update-solver-benchmark-figures:
	@if ls ./preset_results*.json 1> /dev/null 2>&1; then \
		mv -f ./preset_results*.json ./local/docs/data/; \
	fi
	uv run ./local/docs/figures/fig_bm_problems_vectors_and_cons.py ./docs/benchmarks/solver/images --show-plots=false
	uv run ./local/docs/figures/fig_bm_problems_separations.py ./docs/benchmarks/solver/images --show-plots=false
	uv run ./local/docs/figures/fig_bm_problems_preset_results.py ./docs/benchmarks/solver/images ./docs/benchmarks/solver/results --show-plots=false
