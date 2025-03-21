import hydra
import pandas as pd
from datasets import Dataset
from omegaconf import DictConfig
from preprocess import DataCollatorForMultipleChoice, preprocess
from transformers import (
    AutoModelForMultipleChoice,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)


def train_model(model, cfg, tokenizer, tokenized_train_ds):
    training_args = TrainingArguments(
        output_dir=cfg.model.model_dir,
        eval_strategy=cfg.model.eval_strategy,
        save_total_limit=cfg.model.save_total_limit,
        save_strategy=cfg.model.save_strategy,
        load_best_model_at_end=True,
        learning_rate=cfg.model.learning_rate,
        per_device_train_batch_size=cfg.model.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.model.per_device_eval_batch_size,
        num_train_epochs=cfg.model.num_train_epochs,
        weight_decay=cfg.model.weight_decay,
        seed=cfg.model.seed,
        gradient_accumulation_steps=cfg.model.gradient_accumulation_steps,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_ds,
        eval_dataset=tokenized_train_ds,
        tokenizer=tokenizer,
        data_collator=DataCollatorForMultipleChoice(tokenizer=tokenizer),
    )

    trainer.train()
    trainer.save_model(cfg.model.model_dir)


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):

    train_df = pd.read_csv(cfg.data.train_path)
    train_df = train_df[:3]
    train_ds = Dataset.from_pandas(train_df)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_name)

    tokenized_train_ds = train_ds.map(
        lambda example: preprocess(example, tokenizer),
        batched=False,
        num_proc=cfg.tokenization.num_proc,
        remove_columns=cfg.tokenization.remove_columns,
    )
    model = AutoModelForMultipleChoice.from_pretrained(cfg.model.model_name)

    train_model(model, cfg, tokenizer, tokenized_train_ds)

    model.save_pretrained(cfg.model.model_dir)
    tokenizer.save_pretrained(cfg.model.tokenizer_dir)

    print("Training finished!")


if __name__ == "__main__":
    main()
