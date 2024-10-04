import os
import faiss
import json
import pickle
import threading
import time
from transformers import pipeline
from models.transformer import generate_response
from training.fine_tune_dpo import fine_tune_model
from sentence_transformers import SentenceTransformer
import torch
import logging
from util_scripts.knowledge import (
    add_training_knowledge,
    extract_knowledge,
    save_all,
    get_relevant_knowledge,
    add_retrieval_knowledge,
    get_recent_knowledge,
    get_training_knowledge_count,
    load_retrieval_knowledge_store  # {{ New import }}
)

# ---------------------- Configuration ----------------------

# Embedding model for FAISS
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'

# FAISS index files
RETRIEVAL_FAISS_INDEX_FILE = 'retrieval_faiss_index.bin'

# Training knowledge store file
TRAINING_KNOWLEDGE_STORE_FILE = 'training_knowledge_store.json'

# Training pipeline configurations
TRAINING_INTERVAL = 60 * 60  # Train every hour
MIN_TRAINING_SAMPLES = 10    # Minimum samples to trigger training

# Model directories
FINE_TUNED_MODEL_DIR = './fine_tuned_model'

# Add Retrieval Knowledge Store File
RETRIEVAL_KNOWLEDGE_STORE_FILE = 'retrieval_knowledge_store.json'  # {{ New configuration }}

# -----------------------------------------------------------

# Remove or comment out the direct tokenizer and model loading if they are no longer needed
# try:
#     model.train()  # Set model to training mode
#     if torch.cuda.is_available():
#         model.to('cuda')
#     logging.info("Language model loaded successfully.")
# except Exception as e:
#     logging.error(f"Error loading language model: {e}")

# Load the embedding model
logging.info("Loading embedding model...")
try:
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    logging.info("Embedding model loaded successfully.")
except Exception as e:
    logging.error(f"Error loading embedding model: {e}")

# Initialize or load FAISS index for retrieval
if os.path.exists(RETRIEVAL_FAISS_INDEX_FILE):
    logging.info("Loading existing FAISS index for retrieval...")
    try:
        index = faiss.read_index(RETRIEVAL_FAISS_INDEX_FILE)
        logging.info("Existing FAISS index loaded.")
    except Exception as e:
        logging.error(f"Error loading FAISS index: {e}")
else:
    logging.info("Creating a new FAISS index for retrieval...")
    try:
        index = faiss.IndexFlatL2(384)  # Embedding dimension for all-MiniLM-L6-v2 is 384
        logging.info("New FAISS index created.")
    except Exception as e:
        logging.error(f"Error creating FAISS index: {e}")
logging.info("FAISS index for retrieval ready.")

# Initialize or load training knowledge store
if os.path.exists(TRAINING_KNOWLEDGE_STORE_FILE):
    logging.info("Loading existing training knowledge store...")
    try:
        with open(TRAINING_KNOWLEDGE_STORE_FILE, 'r') as f:
            training_knowledge_store = json.load(f)
        logging.info("Training knowledge store loaded.")
    except Exception as e:
        logging.error(f"Error loading training knowledge store: {e}")
else:
    logging.info("Creating a new training knowledge store...")
    training_knowledge_store = []
    logging.info("New training knowledge store created.")
logging.info("Training knowledge store ready.")

# Initialize or load retrieval knowledge store
if os.path.exists(RETRIEVAL_KNOWLEDGE_STORE_FILE):
    logging.info("Loading existing retrieval knowledge store...")
    try:
        retrieval_knowledge_store, retrieval_knowledge_set = load_retrieval_knowledge_store(RETRIEVAL_KNOWLEDGE_STORE_FILE)  # {{ Load both list and set }}
        logging.info("Retrieval knowledge store loaded.")
    except Exception as e:
        logging.error(f"Error loading retrieval knowledge store: {e}")
        retrieval_knowledge_store = []
        retrieval_knowledge_set = set()  # {{ Initialize empty set }}
else:
    logging.info("Creating a new retrieval knowledge store...")
    retrieval_knowledge_store = []
    retrieval_knowledge_set = set()  # {{ Initialize empty set }}
    logging.info("New retrieval knowledge store created.")
logging.info("Retrieval knowledge store ready.")

# ---------------------- Main Function ----------------------

def main():

    # Initialize training trigger event
    training_trigger = threading.Event()

    # Start the training thread with the training_trigger
    training_thread = threading.Thread(
        target=fine_tune_model,
        args=(
            TRAINING_INTERVAL,
            MIN_TRAINING_SAMPLES,
            FINE_TUNED_MODEL_DIR,
            training_knowledge_store,
            training_trigger  # {{ Pass the event to the training thread }}
        ),
        daemon=True
    )
    training_thread.start()

    # Removed model reloading thread
    
    logging.info("Welcome to the Continuous Learning Assistant!")
    logging.info("Type 'exit' to quit.\n")
    
    # Initialize a lock for thread-safe operations
    knowledge_lock = threading.Lock()
    
    # Initialize interaction sequence with empty list
    interaction_sequence = []
    knowledge_history = []
    while True:
        user_input = input("User: ")
        print(f"User input received: {user_input}")
        if user_input.lower() in ['exit', 'quit']:
            logging.info("Exiting. Saving knowledge...")
            save_all(
                index,
                RETRIEVAL_FAISS_INDEX_FILE,
                training_knowledge_store,
                TRAINING_KNOWLEDGE_STORE_FILE,
                retrieval_knowledge_store,
                retrieval_knowledge_set  # {{ Pass the set }}
            )
            break
        knowledge_history.append({"role": "user", "content": user_input})
        print(f"Updated knowledge history: {knowledge_history}")
        
        # Encode user input using the embedding model
        user_embedding = embedding_model.encode([user_input])[0]
        interaction_sequence.append(user_embedding)
        if len(interaction_sequence) > 50:
            interaction_sequence.pop(0)
        print(f"Interaction sequence length: {len(interaction_sequence)}")
        
        # Prepare enhanced user input with embedding
        enhanced_user_input = f"{user_input}\n[Embedding]: {user_embedding.tolist()}"
        system_prompt = "You are a helpful assistant."
        print(f"Initial system prompt: {system_prompt}")
        
        # Retrieve relevant and recent knowledge
        with knowledge_lock:
            recent_knowledge = get_recent_knowledge(retrieval_knowledge_store, limit=10)
            relevant_knowledge = get_relevant_knowledge(
                user_input,
                embedding_model,
                index,
                retrieval_knowledge_store,
                top_k=10
            )
        print(f"Recent knowledge retrieved: {recent_knowledge}")
        print(f"Relevant knowledge retrieved: {relevant_knowledge}")
        
        # Combine recent and relevant knowledge
        combined_knowledge = recent_knowledge + relevant_knowledge
        if combined_knowledge:
            knowledge_text = "\n".join(combined_knowledge)
            system_prompt = f"You are a helpful assistant. You have the following information:\n{knowledge_text}"
            print(f"Updated system prompt with knowledge: {system_prompt}")
        else:
            print("No relevant knowledge found.")
        
        prompt = f"User: {user_input}"
        print(f"Prompt set to: {prompt}")
        
        # Log the final prompts being sent to the model
        print(f"Final System Prompt:\n{system_prompt}")
        print(f"Final User Prompt:\n{prompt}")
        
        # Generate response
        response = generate_response(system_prompt=system_prompt, user_prompt=prompt)
        print(f"Generated response: {response}")
        knowledge_history.append(response)
        print(f"Assistant: {response['content']}\n")
        
        # Collect user feedback
        while True:
            try:
                feedback = int(input("Please rate the response (1-5): "))
                print(f"User feedback received: {feedback}")
                if feedback < 1 or feedback > 5:
                    print("Invalid input. Please enter a number between 1 and 5.")
                    continue
                break
            except ValueError:
                print("Invalid input. Please enter a number between 1 and 5.")
        
        # Collect preferred response based on feedback
        if feedback >= 4:
            preferred_response = None
            semantic_response, episodic_response = extract_knowledge(knowledge_history, "")
            print("No preferred response provided due to positive feedback.")
        else:
            preferred_response = input("Please provide a preferred response: ")
            print(f"Preferred response provided: {preferred_response}")
            semantic_response, episodic_response = extract_knowledge(knowledge_history, preferred_response)

        # Append assistant response embedding to interaction sequence
        assistant_embedding = embedding_model.encode([response['content']])[0]
        interaction_sequence.append(assistant_embedding)
        if len(interaction_sequence) > 50:
            interaction_sequence.pop(0)
        print(f"Interaction sequence updated with assistant embedding. Length: {len(interaction_sequence)}")
        
        # Debug prints
        print(f"response: {response}")
        print(f"preferred_response: {preferred_response}")
        print(f"user_input: {user_input}")
        
        # Prepare knowledge entry
        knowledge = {
            "user_input": user_input,
            "assistant_response": response['content'],
            "preferred_response": preferred_response,
            "semantic_knowledge": semantic_response,
            "episodic_knowledge": episodic_response,
            "feedback": feedback
        }
        print(f"Knowledge entry prepared: {knowledge}")
        
        with knowledge_lock:
            add_training_knowledge(knowledge, training_knowledge_store)
            add_retrieval_knowledge(response['content'], embedding_model, index, retrieval_knowledge_store, retrieval_knowledge_set)  # {{ Pass the set }}
            
            # {{ Add semantic knowledge to retrieval knowledge store }}
            if semantic_response:
                add_retrieval_knowledge(semantic_response, embedding_model, index, retrieval_knowledge_store, retrieval_knowledge_set)
                print("Semantic knowledge added to retrieval knowledge store.")
            
            # {{ Add episodic knowledge to retrieval knowledge store }}
            if episodic_response:
                add_retrieval_knowledge(episodic_response, embedding_model, index, retrieval_knowledge_store, retrieval_knowledge_set)
                print("Episodic knowledge added to retrieval knowledge store.")
            
            logging.info("Interaction added to training knowledge store.")
            logging.info("Retrieval knowledge added to FAISS index and knowledge store.")
        
        # Save knowledge after each interaction
        save_all(
            index,
            RETRIEVAL_FAISS_INDEX_FILE,
            training_knowledge_store,
            TRAINING_KNOWLEDGE_STORE_FILE,
            retrieval_knowledge_store,
            retrieval_knowledge_set  # {{ Pass the set }}
        )
        logging.info("Retrieval FAISS index saved.")
        logging.info("Training knowledge store saved.")
        logging.info("Retrieval knowledge store saved.")
        
        # Real-Time Learning: Handled inherently by updating the interaction sequence and using embeddings
        if get_training_knowledge_count(training_knowledge_store) >= MIN_TRAINING_SAMPLES:
            logging.info("Minimum training samples reached. Triggering model training...")
            training_trigger.set()  # {{ Signal the training thread to start fine-tuning }}

# -----------------------------------------------------------

# ---------------------- Entry Point ----------------------

if __name__ == "__main__":
    main()