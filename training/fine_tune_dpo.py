from transformers import Trainer, TrainingArguments, AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import Dataset, DataLoader
import torch
import json
import logging
import time
import threading  # {{ New import }}
import os  # {{ New import }}


class InteractionDataset(Dataset):
    def __init__(self, interactions, tokenizer, max_length=8192):
        self.inputs = [interaction["user_input"] for interaction in interactions]
        self.targets = [
            interaction["preferred_response"] for interaction in interactions
        ]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input_ids = self.tokenizer.encode(
            self.inputs[idx], truncation=True, max_length=self.max_length
        )
        labels = self.tokenizer.encode(
            self.targets[idx], truncation=True, max_length=self.max_length
        )
        return {"input_ids": torch.tensor(input_ids), "labels": torch.tensor(labels)}


def extract_training_examples(knowledge_store, min_feedback=4):
    """
    Filters interactions based on feedback score.
    Only interactions with feedback >= min_feedback are considered.
    """
    return [
        interaction
        for interaction in knowledge_store
        if interaction.get("feedback", 0) >= min_feedback
    ]


def fine_tune_dpo(
    training_knowledge_store,
    output_dir="./fine_tuned_model",
    epochs=1,
    batch_size=2,
    learning_rate=5e-5,
):
    logging.info("Starting DPO fine-tuning...")
    training_examples = extract_training_examples(training_knowledge_store)
    if not training_examples:
        logging.info("No suitable training examples found for DPO.")
        return

    # Load from fine-tuned model if it exists, else load base model
    if os.path.exists(output_dir) and os.listdir(output_dir):
        logging.info("Loading existing fine-tuned model for further fine-tuning...")
        model = AutoModelForCausalLM.from_pretrained(output_dir)
        tokenizer = AutoTokenizer.from_pretrained(output_dir)
    else:
        logging.info("Loading base model for fine-tuning...")
        model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
    dataset = InteractionDataset(training_examples, tokenizer)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        save_steps=500,
        save_total_limit=2,
        logging_steps=100,
        learning_rate=learning_rate,
        weight_decay=0.01,
        fp16=torch.cuda.is_available(),
    )
    trainer = Trainer(model=model, args=training_args, train_dataset=dataset)

    trainer.train()
    trainer.save_model(output_dir)
    logging.info("DPO fine-tuning completed and model saved locally.")


def fine_tune_model(
    TRAINING_INTERVAL,
    MIN_TRAINING_SAMPLES,
    FINE_TUNED_MODEL_DIR,
    training_knowledge_store,
    training_trigger,
):  # {{ Added training_trigger parameter }}
    """
    Periodically fine-tunes the model using accumulated training examples.
    Runs in a separate daemon thread.
    """
    while True:
        # Wait for the training trigger to be set
        training_trigger.wait()  # {{ Wait until triggered }}
        logging.info("Training trigger received. Starting fine-tuning.")

        if len(training_knowledge_store) < MIN_TRAINING_SAMPLES:
            logging.info("Not enough training samples. Waiting for more interactions.")
            training_trigger.clear()  # {{ Reset the trigger }}
            continue

        fine_tune_dpo(training_knowledge_store, output_dir=FINE_TUNED_MODEL_DIR)
        # Clear the training knowledge store after training
        training_knowledge_store.clear()
        logging.info("Training knowledge store cleared after fine-tuning.")

        training_trigger.clear()  # {{ Reset the trigger }}

        # Optionally, sleep for the TRAINING_INTERVAL before next check
        time.sleep(TRAINING_INTERVAL)
