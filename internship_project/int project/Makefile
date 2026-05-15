SHELL := /bin/bash
PYTHON ?= python
APP := app.py
PID_FILE := .server.pid
LOG_FILE := server.log

.PHONY: start stop status

start:
	@if [[ -f $(PID_FILE) ]] && kill -0 $$(cat $(PID_FILE)) 2>/dev/null; then \
		echo "Server already running (PID $$(cat $(PID_FILE)))."; \
	else \
		echo "Starting server..."; \
		nohup $(PYTHON) $(APP) > $(LOG_FILE) 2>&1 & echo $$! > $(PID_FILE); \
		echo "Server started (PID $$(cat $(PID_FILE)))."; \
		echo "Logs: $(LOG_FILE)"; \
	fi

stop:
	@if [[ -f $(PID_FILE) ]]; then \
		PID=$$(cat $(PID_FILE)); \
		if kill -0 $$PID 2>/dev/null; then \
			kill $$PID; \
			echo "Stopped server (PID $$PID)."; \
		else \
			echo "No running server found for PID $$PID."; \
		fi; \
		rm -f $(PID_FILE); \
	else \
		echo "No PID file found."; \
	fi

status:
	@if [[ -f $(PID_FILE) ]] && kill -0 $$(cat $(PID_FILE)) 2>/dev/null; then \
		echo "Server running (PID $$(cat $(PID_FILE)))."; \
	else \
		echo "Server not running."; \
	fi
