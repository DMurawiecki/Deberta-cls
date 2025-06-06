from dataclasses import dataclass
from typing import Optional, Union

import pandas as pd
import pytorch_lightning as pl
import torch
from datasets import Dataset
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from transformers.tokenization_utils_base import PaddingStrategy



def preprocess(example, tokenizer):
    # The AutoModelForMultipleChoice class expects a set of question/answer pairs
    # so we'll copy our question 5 times before tokenizing
    options = "ABCDE"
    indices = list(range(5))

    option_to_index = {option: index for option, index in zip(options, indices)}
    # index_to_option = {index: option for option, index in zip(options, indices)}

    first_sentence = [example["prompt"]] * 5
    second_sentence = []
    for option in options:
        second_sentence.append(example[option])
    # Our tokenizer will turn our text into token IDs BERT can understand
    tokenized_example = tokenizer(first_sentence, second_sentence, truncation=True)
    tokenized_example["label"] = option_to_index[example["answer"]]
    return tokenized_example


@dataclass
class DataCollatorForMultipleChoice:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features):
        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = [feature.pop(label_name) for feature in features]
        batch_size = len(features)
        num_choices = len(features[0]["input_ids"])

        flattened_features = []
        for feature in features:
            for i in range(num_choices):
                flattened_features.append(
                    {
                        k: v[i]
                        for k, v in feature.items()
                        if isinstance(v, (list, tuple))
                    }
                )

        batch = self.tokenizer.pad(
            flattened_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
        batch["labels"] = torch.tensor(labels, dtype=torch.int64)
        return batch


class MultipleChoiceDataModule(pl.LightningDataModule):
    def __init__(self, cfg: DictConfig, tokenizer=None):
        super().__init__()
        self.cfg = cfg
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(
            cfg.model.tokenizer_dir
        )
        self.tokenized_train_ds = None
        self.tokenized_test_ds = None

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            train_df = pd.read_csv(self.cfg.data.train_path)
            # train_df = train_df[:3]
            train_ds = Dataset.from_pandas(train_df)

            self.tokenized_train_ds = train_ds.map(
                lambda example: preprocess(example, self.tokenizer),
                batched=False,
                num_proc=self.cfg.tokenization.num_proc,
                remove_columns=self.cfg.tokenization.remove_columns,
            )

        if stage in ["test", "predict"] or stage is None:
            test_df = pd.read_csv(self.cfg.data.test_path)
            test_df["answer"] = "A"
            test_df = test_df[:3]
            test_ds = Dataset.from_pandas(test_df)

            self.tokenized_test_ds = test_ds.map(
                lambda example: preprocess(example, self.tokenizer),
                batched=False,
                remove_columns=self.cfg.tokenization.remove_columns,
            )

    def train_dataloader(self):
        return DataLoader(
            self.tokenized_train_ds,
            batch_size=self.cfg.model.per_device_train_batch_size,
            shuffle=True,
            collate_fn=DataCollatorForMultipleChoice(tokenizer=self.tokenizer),
        )

    def val_dataloader(self):
        return DataLoader(
            self.tokenized_train_ds,
            batch_size=self.cfg.model.per_device_eval_batch_size,
            shuffle=False,
            collate_fn=DataCollatorForMultipleChoice(tokenizer=self.tokenizer),
        )

    def predict_dataloader(self):
        return DataLoader(
            self.tokenized_test_ds,
            batch_size=self.cfg.model.per_device_eval_batch_size,
            shuffle=False,
            collate_fn=DataCollatorForMultipleChoice(tokenizer=self.tokenizer),
        )
