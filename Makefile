file_path=

.PHONY: help build test coverage format format-single-file splash docs show-coverage show-docs

help:
	@echo 'Commands:'
	@echo ''
	@echo '  help                           Show this help message.'
	@echo ''
	@echo '  build                          (Re)build package using uv.'
	@echo ''
	@echo '  test                           Run pytest unit tests.'
	@echo '  format                         Format source code using ruff.'
	@echo '  format-single-file             Format single file using ruff. Useful in e.g. PyCharm to automatically trigger formatting on file save.'
	@echo ''
	@echo '  splash                         Build splash screen using current version of package. (./images/splash_with_version.webp)'
	@echo '  coverage                       Generate test coverage report. (./reports/coverage)'
	@echo '  docs                           Update mkdocs site. (./reports/docs)'
	@echo ''
	@echo '  show-coverage                  Open test coverage report in browser.'
	@echo '  show-docs                      Open mkdocs site in browser.'
	@echo ''
	@echo '  update-internal-benchmarks     Run internal benchmarks and update results in ./docs/benchmarks/internal/results/.'
	@echo '  update-solver-benchmarks       Run solver benchmarks and update results in ./docs/benchmarks/solver/results/.'
	@echo '  update-all-benchmarks          Run all benchmarks and update results in ./docs/benchmarks/.'
	@echo ''
	@echo 'Options:'
	@echo ''
	@echo '  format-single-file             - accepts `file_path=<path>` to pass the relative path of the file to be formatted.'

build:
	uv build;

test:
	# run all tests - with numba & just 1 python version
	uv run --all-extras --python 3.13 pytest ./tests --durations=20 --disable-warnings

coverage:
	# NOTE: NUMBA_DISABLE_JIT ensure coverage collects detailed line-by-line coverage info, also for numba-compiled functions
    #       NUMBA_JIT_COVERAGE is another option, but would incorrectly emit coverage info for ALL compiled lines, when a function is triggered.
	mkdir -p ./reports
	# run tests with Python 3.14; WITH ALL optional dependencies & append to report
	NUMBA_DISABLE_JIT=1 COVERAGE_FILE=./reports/.coverage uv run --all-extras --python 3.13 pytest ./tests --cov --cov-report=html:./reports/coverage --durations=20 --disable-warnings

format:
	uvx ruff format .;
	uvx ruff check --fix .;

format-single-file:
	uvx ruff format ${file_path};
	uvx ruff check --fix ${file_path};

splash:
	./images/splash/create_splash_with_version.sh "$$(uv version --short)-dev";

docs:
	mkdir -p ./reports
	uv sync --extra docs;
	mkdocs build;

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

update-solver-benchmarks:
	uv run max-div benchmark solver run --markdown --file all;
	rm ./docs/benchmarks/solver/results/*.md;
	mv ./benchmark*.md ./docs/benchmarks/solver/results/;
	$(MAKE) docs

update-all-benchmarks: update-internal-benchmarks update-solver-benchmarks