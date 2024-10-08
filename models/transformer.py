import logging
from transformers import pipeline
import torch
import os

# Transformer model selection
MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"  # Replace with desired model

# Model directories
FINE_TUNED_MODEL_DIR = "./fine_tuned_model"

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(message)s")

# Load the language model and tokenizer
logging.info("Loading language model...")
try:

    # Load the fine-tuned model if it exists, otherwise load the base model
    if os.path.exists(FINE_TUNED_MODEL_DIR) and os.listdir(FINE_TUNED_MODEL_DIR):
        logging.info("Loading fine-tuned model...")
        MODEL_NAME = FINE_TUNED_MODEL_DIR
    else:
        logging.info("Loading base Transformers model...") 
        
    # Initialize the text-generation pipeline
    pipe = pipeline(
        "text-generation",
        model=MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
    )
    logging.info("Language model pipeline loaded successfully.")
except Exception as e:
    logging.error(f"Error loading language model pipeline: {e}")


def generate_response(system_prompt, user_prompt, max_length=4096):
    logging.debug(f"Generating response for prompt: {user_prompt}")
    try:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        outputs = pipe(
            messages,
            max_new_tokens=max_length,
        )
        response = outputs[0]["generated_text"][-1]
        logging.debug(f"Generated response: {response}")
        return response
    except Exception as e:
        logging.error(f"Error generating response: {e}")
        return "I'm sorry, I encountered an error while generating a response."
