# Define variables
VENV = venv
PYTHON = python3
FLASK_APP = app.py
PIP = $(VENV)/bin/pip
FLASK = $(VENV)/bin/flask

.PHONY: install run clean reinstall

# Create virtual environment and install dependencies
install:
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

# Run the Flask application
run:
	@if [ ! -d "$(VENV)" ]; then \
		echo "Virtual environment not found. Running install first..."; \
		make install; \
	fi
	FLASK_APP=$(FLASK_APP) FLASK_ENV=development $(FLASK) run --port 3000

# Clean up virtual environment
clean:
	rm -rf $(VENV)

# Reinstall all dependencies
reinstall: clean install