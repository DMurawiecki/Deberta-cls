import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig
from transformers import AutoModelForMultipleChoice, AutoTokenizer

from pl_data import MultipleChoiceDataModule
from pl_model import MultipleChoiceLightningModule


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    pl.seed_everything(cfg.model.seed)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_name)
    model = AutoModelForMultipleChoice.from_pretrained(cfg.model.model_name)

    lightning_model = MultipleChoiceLightningModule(cfg, model, tokenizer)

    dm = MultipleChoiceDataModule(cfg, tokenizer)

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

    logger = pl.loggers.MLFlowLogger(
        experiment_name="Deberta-cls",
        run_name="current_run",
        save_dir=".",
        tracking_uri="http://127.0.0.1:8080",
    )

    # logger = None
    # callbacks = None

    trainer = pl.Trainer(
        max_epochs=cfg.model.num_train_epochs,
        accelerator="auto",
        devices="auto",
        callbacks=callbacks,
        logger=logger,
    )

    trainer.fit(lightning_model, datamodule=dm)

    lightning_model.model.save_pretrained(cfg.model.model_dir)
    tokenizer.save_pretrained(cfg.model.tokenizer_dir)

    print("Training finished!")


if __name__ == "__main__":
    main()
