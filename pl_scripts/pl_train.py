import os

import fire
import pytorch_lightning as pl
from hydra import compose, initialize
from omegaconf import DictConfig
from pl_data import MultipleChoiceDataModule
from pl_model import MultipleChoiceLightningModule
from transformers import AutoModelForMultipleChoice, AutoTokenizer


def train(cfg: DictConfig, logger: str | None = None):
    pl.seed_everything(cfg.model.seed)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_name)
    model = AutoModelForMultipleChoice.from_pretrained(cfg.model.model_name)
    lightning_model = MultipleChoiceLightningModule(cfg, model, tokenizer)
    dm = MultipleChoiceDataModule(cfg, tokenizer)

    if logger == "mlflow":
        callbacks = [
            pl.callbacks.LearningRateMonitor(logging_interval="step"),
            pl.callbacks.ModelCheckpoint(
                dirpath=cfg.model.model_dir,
                filename="{epoch:02d}-{val_loss:.4f}",
                monitor="val_loss",
                save_top_k=cfg.model.get("save_top_k", 1),
                every_n_epochs=cfg.model.get("every_n_epochs", 1),
            ),
        ]
        logger_obj = pl.loggers.MLFlowLogger(
            experiment_name="Sci-checked-bot",
            run_name="current_run",
            save_dir=".",
            tracking_uri="http://127.0.0.1:8080",
        )
    else:
        callbacks = None
        logger_obj = None

    trainer = pl.Trainer(
        max_epochs=cfg.model.num_train_epochs,
        accelerator="cpu",
        devices="auto",
        callbacks=callbacks,
        logger=logger_obj,
        strategy="ddp",
    )
    trainer.fit(lightning_model, datamodule=dm)
    lightning_model.model.save_pretrained(cfg.model.model_dir)
    tokenizer.save_pretrained(cfg.model.tokenizer_dir)
    print("Training finished!")


def run(
    config_name: str = "config",
    overrides: str | list[str] | None = None,
    logger: str | None = None,
):
    if isinstance(overrides, str):
        overrides = [o.strip() for o in overrides.split(",") if o.strip()]

    overrides = overrides or []

    with initialize(config_path="../configs", job_name="app"):
        cfg = compose(config_name=config_name, overrides=overrides)
    train(cfg, logger=logger)


if __name__ == "__main__":
    os.environ["HYDRA_FULL_ERROR"] = "1"
    fire.Fire(run)

# python pl_train.py \
#   --config_name=config \
#   --overrides=model.seed=42,model.num_train_epochs=10 \
#   --logger=mlflow
