.PHONY: install format lint test run clean

# Install all dev dependencies
install:
	pip install -r dev-requirements.txt

# Format code automatically
format:
	black atlas_gateway forecasting examples
	isort atlas_gateway forecasting examples

# Check code style (no modifications)
lint:
	flake8 atlas_gateway forecasting examples
	black --check atlas_gateway forecasting examples
	isort --check-only atlas_gateway forecasting examples

# Run tests
test:
	pytest -q --disable-warnings --maxfail=1

# Run the gateway locally
run:
	uvicorn atlas_gateway.main:app --reload --port 8080

# Clean caches
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete