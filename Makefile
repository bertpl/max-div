file_path=

.PHONY: help build test coverage test-and-coverage format format-single-file lint dev-setup release splash docs show-coverage show-docs update-internal-benchmarks update-solver-strategies-benchmarks update-all-benchmarks update-solver-benchmark-figures

help:
	@echo 'Commands:'
	@echo ''
	@echo '  help                                   Show this help message.'
	@echo ''
	@echo '  build                                  (Re)build package using uv.'
	@echo ''
	@echo '  test                                   Run pytest unit tests.'
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

build:
	uv build;

test:
	# run all tests - with numba & just 1 python version
	uv run --all-extras --python 3.13 pytest ./tests --durations=20 --disable-warnings

test-benchmarks:
	# comparison-benchmark harness tests - separate from the package suite (needs the benchmarks deps groups)
	uv run --group benchmarks --group benchmarks-gpl --python 3.13 pytest ./benchmarks/tests --durations=20 --disable-warnings

coverage:
	# NOTE: NUMBA_DISABLE_JIT ensure coverage collects detailed line-by-line coverage info, also for numba-compiled functions
    #       NUMBA_JIT_COVERAGE is another option, but would incorrectly emit coverage info for ALL compiled lines, when a function is triggered.
	mkdir -p ./reports
	# run tests with Python 3.14; WITH ALL optional dependencies & append to report
	NUMBA_DISABLE_JIT=1 COVERAGE_FILE=./reports/.coverage uv run --all-extras --python 3.13 pytest ./tests --cov --cov-report=html:./reports/coverage --durations=20 --disable-warnings

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
