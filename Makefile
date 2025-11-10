file_path=

help:
	@echo 'Commands:'
	@echo ''
	@echo '  help		                    Show this help message.'
	@echo ''
	@echo '  build		                    (Re)build package using uv.'
	@echo ''
	@echo '  test		                    Run pytest unit tests.'
	@echo '  format		                    Format source code using ruff.'
	@echo '  format-single-file             Format single file using ruff. Useful in e.g. PyCharm to automatically trigger formatting on file save.'
	@echo ''
	@echo '  splash       			        Build splash screen using current version of package.'
	@echo ''
	@echo '  mkdocs       			        Update mkdocs site.'
	@echo ''
	@echo 'Options:'
	@echo ''
	@echo '  format-single-file             - accepts `file_path=<path>` to pass the relative path of the file to be formatted.'

build:
	uv build;

test:
	# run all tests - with numba & just 1 python version
	uv run --all-extras --python 3.13 pytest ./tests

coverage:
	# NOTE: NUMBA_DISABLE_JIT ensure coverage collects detailed line-by-line coverage info, also for numba-compiled functions
    #       NUMBA_JIT_COVERAGE is another option, but would incorrectly emit coverage info for ALL compiled lines, when a function is triggered.
	# run tests with Python 3.11; WITHOUT optional dependencies & create new report
	uv sync	# should remove numba
	NUMBA_DISABLE_JIT=1 uv run --python 3.11 pytest ./tests --cov --cov-report=html
	# run tests with Python 3.13; WITH ALL optional dependencies & append to report
	NUMBA_DISABLE_JIT=1 uv run --all-extras --python 3.13 pytest ./tests --cov --cov-append --cov-report=html

format:
	uvx ruff format .;
	uvx ruff check --fix .;

format-single-file:
	uvx ruff format ${file_path};
	uvx ruff check --fix ${file_path};

splash:
	./images/splash/create_splash_with_version.sh "$$(uv version --short)-dev";

mkdocs:
	uv sync --extra docs;
	mkdocs build;