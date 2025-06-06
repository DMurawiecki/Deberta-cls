import os

import hydra
import numpy as np
import pandas as pd
from datasets import Dataset
from omegaconf import DictConfig
from preprocess import DataCollatorForMultipleChoice, preprocess
from transformers import AutoModelForMultipleChoice, AutoTokenizer, Trainer

index_to_option = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E"}

def predictions_to_map_output(predictions):
    sorted_answer_indices = np.argsort(-predictions)
    top_answer_indices = sorted_answer_indices[
        :, :3
    ]  # Get the first three answers in each row
    top_answers = np.vectorize(index_to_option.get)(top_answer_indices)
    return np.apply_along_axis(lambda row: " ".join(row), 1, top_answers)


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    test_df = pd.read_csv(cfg.data.test_path)
    test_df["answer"] = "A"
    test_df = test_df[:3]
    test_ds = Dataset.from_pandas(test_df)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.tokenizer_dir)
    model = AutoModelForMultipleChoice.from_pretrained(cfg.model.model_dir)
    tokenized_test_ds = test_ds.map(
        lambda example: preprocess(example, tokenizer),
        batched=False,
        remove_columns=cfg.tokenization.remove_columns,
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=DataCollatorForMultipleChoice(tokenizer=tokenizer),
    )

    predictions = trainer.predict(tokenized_test_ds)

    output_array = predictions_to_map_output(predictions.predictions)
    output_df = pd.DataFrame({"prediction": output_array})
    output_df.to_csv(cfg.data.output_path, index=False)


if __name__ == "__main__":
    os.environ["HYDRA_FULL_ERROR"] = "1"
    print(os.getcwd())
    main()
