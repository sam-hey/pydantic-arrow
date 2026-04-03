.PHONY: install test clean

SRC_DIR = ./src/pydantic_arrow
TEST_DIR= ./tests
VENV = ./.venv

build: install
	uv build

$(VENV)/.venv_created:
	@command -v uv >/dev/null 2>&1 || { echo "Error: uv is not installed. Please install it first."; exit 1; }
	@uv venv $(VENV)
	@touch $(VENV)/.venv_created

install: $(VENV)/.venv_created
	uv sync --all-extras
	uv run pre-commit install

test: install
	uv run pytest $(TEST_DIR) -n auto --cov-branch --cov-report term --cov-report html:reports --cov-fail-under=90  --cov=$(SRC_DIR) --memray

lint: install
	@echo "🔍 Running lint checks..."
	uv run pre-commit run --all-files

clean:
	@rm -rf __pycache__ $(SRC_DIR)/*.egg-info **/__pycache__ .pytest_cache
	@rm -rf .coverage reports dist
