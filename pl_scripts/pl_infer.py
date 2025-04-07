import os
from typing import List

import hydra
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from omegaconf import DictConfig
from transformers import AutoModelForMultipleChoice, AutoTokenizer

from pl_data import MultipleChoiceDataModule
from pl_model import MultipleChoiceLightningModule

index_to_option = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E"}


def predictions_to_map_output(predictions: List[np.ndarray]):
    all_predictions = np.concatenate(predictions)
    sorted_answer_indices = np.argsort(-all_predictions)
    top_answer_indices = sorted_answer_indices[
        :, :3
    ]  # Get the first three answers in each row
    top_answers = np.vectorize(index_to_option.get)(top_answer_indices)
    return np.apply_along_axis(lambda row: " ".join(row), 1, top_answers)


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.tokenizer_dir)
    dm = MultipleChoiceDataModule(cfg, tokenizer)
    dm.setup(stage="predict")

    model = AutoModelForMultipleChoice.from_pretrained(cfg.model.model_dir)
    lightning_model = MultipleChoiceLightningModule(cfg, model, tokenizer)

    trainer = pl.Trainer(accelerator="auto", devices="auto")

    predictions = trainer.predict(lightning_model, datamodule=dm)

    output_array = predictions_to_map_output(predictions)
    output_df = pd.DataFrame({"prediction": output_array})
    output_df.to_csv(cfg.data.output_path, index=False)


if __name__ == "__main__":
    os.environ["HYDRA_FULL_ERROR"] = "1"
    print(os.getcwd())
    main()
