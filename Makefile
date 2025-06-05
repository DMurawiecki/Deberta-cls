SHELL              := /usr/bin/env bash

PYTHON             ?= python3.11
POETRY             ?= poetry

MLFLOW_HOST        ?= 127.0.0.1
MLFLOW_PORT        ?= 8080
MLFLOW_BACKEND     ?= sqlite:///mlflow.db
MLFLOW_ARTIFACT_ROOT ?= ./mlruns
TRACKING_URI       := http://$(MLFLOW_HOST):$(MLFLOW_PORT)

define ensure_mlflow_running
	@echo "→ Checking MLflow on $(TRACKING_URI)…"
	@lsof -i:$(MLFLOW_PORT) >/dev/null 2>&1 || ( \
		$(MAKE) -s mlflow & \
		sleep 5 \
	)
endef

.PHONY: setup activate clean mlflow download_data train infer run_bot

# -----------------------------------------------------------------------------
# 1) Environment setup (Poetry + dependencies)
# -----------------------------------------------------------------------------
setup:
	@echo "🍿  Installing Poetry if necessary…"
	@command -v $(POETRY) >/dev/null 2>&1 || (curl -sSL https://install.python-poetry.org | $(PYTHON))
	@echo "🐍  Using Python: $(PYTHON)"
	@$(POETRY) env use $(PYTHON)
	@echo "📦  Installing dependencies…"
	@$(POETRY) install --no-root

activate:
	@$(POETRY) shell

clean:
	@echo "🧹  Removing virtual environment…"
	@$(POETRY) env remove --all || true

# -----------------------------------------------------------------------------
# 2) MLflow Tracking Server
# -----------------------------------------------------------------------------
mlflow:
	@echo "🚀  Starting MLflow tracking server at $(TRACKING_URI)…"
	@$(POETRY) run mlflow server \
	    --backend-store-uri $(MLFLOW_BACKEND) \
	    --default-artifact-root $(MLFLOW_ARTIFACT_ROOT) \
	    --host $(MLFLOW_HOST) --port $(MLFLOW_PORT)

# -----------------------------------------------------------------------------
# 3) Data download helper
# -----------------------------------------------------------------------------
download: setup
	@echo "⬇️   Downloading data…"
	@$(POETRY) run python download_data.py
	@echo "✅  Data downloaded successfully!"

# -----------------------------------------------------------------------------
# 4) Training
# -----------------------------------------------------------------------------
train: setup
	$(call ensure_mlflow_running)
	@echo "🏋️   Training model…"
	@MLFLOW_TRACKING_URI=$(TRACKING_URI) $(POETRY) run python pl_scripts/pl_train.py \
		--config_name=config \
		--overrides=model.seed=42,model.num_train_epochs=10 \
		--logger=mlflow
	@echo "🎉  Model trained successfully!"

# -----------------------------------------------------------------------------
# 5) Inference
# -----------------------------------------------------------------------------
infer: setup
	$(call ensure_mlflow_running)
	@echo "🔎  Running inference…"
	@MLFLOW_TRACKING_URI=$(TRACKING_URI) $(POETRY) run python pl_scripts/pl_infer.py
	@echo "✅  Inference complete!"

# -----------------------------------------------------------------------------
# 6) Bot runner (placeholder — update command as needed)
# -----------------------------------------------------------------------------
run_bot:
	@echo "🤖  Starting bot…"
	@docker-compose up -d

down_bot:
	@echo "🛑  Stopping and removing containers…"
	@docker-compose down
