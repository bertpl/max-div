# --- how the package suite is installed and run ------------
# Single definition, shared by every target below AND by the CI test matrix, which calls these
# targets rather than repeating the command. That is the point: the interpreter, the resolution
# strategy and the dependency group cannot drift between a developer machine and CI, because
# there is only one place that names them. CI overrides PY and RESOLUTION per matrix leg; the
# defaults are the single representative combo to run locally.
#
# On the uv flags. Every `uv run` / `uv sync` implicitly activates a set of dependency groups —
# uv's "default groups", which is `dev` unless a project configures otherwise. `--group` ADDS to
# that set rather than replacing it, so `--group test` on its own installs test AND dev, and the
# narrowing would be cosmetic. `--no-default-groups` empties the implicit set, leaving only what
# is named. `--only-group` narrows as well, but additionally drops the project and its runtime
# dependencies, which leaves pytest with nothing to import.
PY ?= 3.14
RESOLUTION ?= highest
UV_RUN = uv run --exact --python $(PY) --resolution $(RESOLUTION) --no-default-groups --group test

# Appended to the pytest invocation. Locally this silences the warning summary; CI overrides it
# to pass coverage flags, and deliberately keeps the warnings visible.
PYTEST_ARGS ?= --disable-warnings

.PHONY: help build test collect-test-ids check-capability-data build-capability-data coverage test-and-coverage lint dev-setup release splash docs show-coverage show-docs update-internal-benchmarks update-solver-strategies-benchmarks update-solver-feasibility-benchmarks update-all-benchmarks update-solver-benchmark-figures

help:
	@echo 'Commands:'
	@echo ''
	@echo '  help                                   Show this help message.'
	@echo ''
	@echo '  build                                  (Re)build package using uv.'
	@echo ''
	@echo '  test                                   Run pytest unit tests.'
	@echo '  collect-test-ids                       List collected test node-ids without running them.'
	@echo '  check-capability-data                  Validate the solver capability data without writing anything.'
	@echo '  build-capability-data                  Validate it and regenerate the feature tables.'
	@echo '  coverage                               Generate test coverage report. (./reports/coverage)'
	@echo '  test-and-coverage                      Run unit tests (=with numba) + generate coverage report (=without numba).'
	@echo ''
	@echo '  lint                                   Run all pre-commit hooks on all files. Formats and applies ruff fixes as it goes.'
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
	@echo '  update-solver-feasibility-benchmarks   Run solver feasibility benchmarks and update results in ./docs/benchmarks/solver/results/.'
	@echo '  update-all-benchmarks                  Run all benchmarks and update results in ./docs/benchmarks/.'
	@echo '  update-solver-benchmark-figures        Updates all benchmark-related figures in ./docs/benchmarks/solver/images.'
	@echo ''
	@echo 'Options:'
	@echo ''
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

# The capability data behind the comparison surfaces. `check` is the dry run to reach for while
# authoring a record — it reports the same problems the pre-commit hook and CI would, at the point
# where the record is still fresh in mind.
#
# The README hero SVGs are build products of the same records, so both targets cover them too: a
# record edit that leaves the images behind is the same drift, whichever surface shows it first.
check-capability-data:
	uv run --no-default-groups --group tooling python scripts/capability_data.py --check
	uv run --no-default-groups --group tooling python scripts/build_hero_table.py --check

build-capability-data:
	uv run --no-default-groups --group tooling python scripts/capability_data.py
	uv run --no-default-groups --group tooling python scripts/build_hero_table.py

test-benchmarks:
	# comparison-benchmark harness tests - separate from the package suite (needs the benchmarks deps group)
	uv run --group benchmarks --python 3.14 pytest ./benchmarks/tests --durations=20 --disable-warnings

coverage:
	# NOTE: NUMBA_DISABLE_JIT ensure coverage collects detailed line-by-line coverage info, also for numba-compiled functions
    #       NUMBA_JIT_COVERAGE is another option, but would incorrectly emit coverage info for ALL compiled lines, when a function is triggered.
	mkdir -p ./reports
	# same install surface as `test` and as the CI coverage legs (see PY / RESOLUTION above)
	NUMBA_DISABLE_JIT=1 COVERAGE_FILE=./reports/.coverage $(UV_RUN) pytest ./tests --cov --cov-report=html:./reports/coverage --durations=20 $(PYTEST_ARGS)

test-and-coverage:
	$(MAKE) test;
	$(MAKE) coverage;

lint:
	uv run --exact --no-default-groups --group lint pre-commit run --all-files

dev-setup:
	uv sync --all-extras
	uv run pre-commit install

release:
ifndef VERSION
	$(error VERSION is required: make release VERSION=X.Y.Z)
endif
	# preconditions first (seconds, and step 8 waits out an in-flight main CI run), so a bad
	# changelog or a stale coverage run aborts before the test suite spends its minutes
	uv run python scripts/release.py $(VERSION) --dry-run
	$(MAKE) test
	uv run python scripts/release.py $(VERSION)

splash:
	./images/splash/create_splash_with_version.sh "$$(uv version --short)-dev";

docs:
	# output dir comes from mkdocs.yml (site_dir: ./reports/docs)
	uv run --exact --no-default-groups --group docs mkdocs build --strict;

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

update-solver-feasibility-benchmarks:
	uv run max-div benchmark solver feasibility --markdown --file --problem=all;
	mv -f ./feasibility_verdicts_*.md ./docs/benchmarks/solver/results/;
	$(MAKE) docs

update-all-benchmarks: update-internal-benchmarks update-solver-strategies-benchmarks update-solver-feasibility-benchmarks

update-solver-benchmark-figures:
	@if ls ./preset_results*.json 1> /dev/null 2>&1; then \
		mv -f ./preset_results*.json ./local/docs/data/; \
	fi
	# Geometry: a self-contained script (no local.docs imports), so a plain-path run under the
	# benchmarks group for matplotlib.
	uv run --group benchmarks ./scripts/generate_problem_images.py
	# The two local.docs generators run as modules so their package imports resolve from the repo
	# root, under the notebooks group whose cvxpy their local.docs.utils package init pulls in.
	uv run --group notebooks python -m local.docs.figures.fig_bm_problems_separations ./docs/benchmarks/solver/images --show-plots=false
	uv run --group notebooks python -m local.docs.figures.fig_bm_problems_preset_results ./docs/benchmarks/solver/images ./docs/benchmarks/solver/results --show-plots=false
