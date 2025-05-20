# Makefile for training a model and running a bot

# Variables
MODEL_FILE = model.pth
BOT_SCRIPT = run_bot.py
TRAIN_SCRIPT = train_model.py

# Targets
all: train bot

train: $(MODEL_FILE)

$(MODEL_FILE): $(TRAIN_SCRIPT)
	python $(TRAIN_SCRIPT)

bot: $(MODEL_FILE)
	python $(BOT_SCRIPT)

clean:
	rm -f $(MODEL_FILE)
