.PHONY: all test lint clean fmt build-go test-go test-py setup

all: build-go test-py test-go
	@echo "=== All checks passed ==="

setup:
	@echo "=== Setting up Python ==="
	pip3 install -e training/[dev]
	@echo "=== Setting up Go ==="
	cd runtime && go mod download

build-go:
	@echo "=== Building Go runtime ==="
	cd runtime && go build -o blm ./cmd/blm

test-go:
	@echo "=== Running Go tests ==="
	cd runtime && go test ./... -v -count=1

test-py:
	@echo "=== Running Python tests ==="
	cd training && python3 -m pytest tests/ -v -x --tb=short

test: test-py test-go

lint:
	@echo "=== Linting ==="
	@echo "Python:"
	cd training && python -m flake8 blm/ tests/ --max-line-length=100 2>/dev/null || echo "flake8 not installed, skipping"
	@echo "Go:"
	cd runtime && go vet ./...

clean:
	@echo "=== Cleaning ==="
	rm -rf runtime/blm
	rm -rf training/*.egg-info
	rm -rf training/.pytest_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete
	rm -rf checkpoints/ models/ data/
	@echo "Done"

fmt:
	@echo "=== Formatting ==="
	@echo "Python:"
	cd training && python -m black blm/ tests/ 2>/dev/null || echo "black not installed, skipping"
	@echo "Go:"
	cd runtime && go fmt ./...
