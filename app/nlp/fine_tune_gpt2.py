from pathlib import Path
import math

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

MODEL_NAME = "openai-community/gpt2"

ANSWER_PREFIX = "That is a great question. "
ANSWER_SUFFIX = " Let me know if you have any other questions."

MAX_LENGTH = 256
OUTPUT_DIR = "models/gpt2-squad-formatted"


def build_example(example):
    question = example["question"]
    context = example["context"]
    answer = example["answers"]["text"][0]

    prompt = (
        f"Question: {question}\n"
        f"Context: {context}\n"
        f"Answer: {ANSWER_PREFIX}"
    )
    target = f"{answer}{ANSWER_SUFFIX}"

    full_text = prompt + target
    return {"text": full_text}


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading SQuAD dataset...")
    raw_datasets = load_dataset("rajpurkar/squad")

    # train
    max_train_samples = 1000
    train_dataset = raw_datasets["train"].select(range(max_train_samples))

    processed_train = train_dataset.map(
        build_example,
        remove_columns=train_dataset.column_names,
    )

    def tokenize_function(examples):
        outputs = tokenizer(
            examples["text"],
            truncation=True,
            max_length=MAX_LENGTH,
            padding="max_length",
        )
        # causal LM: labels = input_ids
        outputs["labels"] = outputs["input_ids"].copy()
        return outputs

    tokenized_train = processed_train.map(
        tokenize_function,
        batched=True,
        remove_columns=processed_train.column_names,
    )

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model.resize_token_embeddings(len(tokenizer))

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        overwrite_output_dir=True,
        num_train_epochs=1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=5e-5,
        weight_decay=0.01,
        warmup_steps=100,
        logging_steps=50,
        save_steps=500,
        save_total_limit=2,
        fp16=False,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        data_collator=data_collator,
    )

    print("Start training...")
    train_result = trainer.train()
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    metrics = train_result.metrics
    try:
        metrics["perplexity"] = math.exp(metrics["train_loss"])
    except OverflowError:
        metrics["perplexity"] = float("inf")

    print("Training finished. Metrics:", metrics)
    print(f"Model saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    main()