import json
import logging
import faiss
from models.transformer import pipe
import threading  # {{ New import }}

# Initialize a lock for thread-safe operations
knowledge_lock = threading.Lock()  # {{ New addition }}


# Load prompt files
def load_prompt(file_path):
    with open(file_path, "r") as file:
        return file.read()


# Load semantic and episodic knowledge prompts
SEMANTIC_PROMPT = load_prompt("prompts/semantic_knowledge.txt")
EPISODIC_PROMPT = load_prompt("prompts/episodic_knowledge.txt")

# Add Retrieval Knowledge Store File Path
RETRIEVAL_KNOWLEDGE_STORE_FILE = "retrieval_knowledge_store.json"


def add_retrieval_knowledge(
    text, embedding_model, index, retrieval_knowledge_store, retrieval_knowledge_set
):
    logging.debug(f"Adding retrieval knowledge: {text}")
    try:
        with knowledge_lock:  # {{ Ensure thread safety }}
            # Check for duplication using a set for O(1) lookup
            if text in retrieval_knowledge_set:
                logging.debug("Knowledge already exists. Skipping addition.")
                return
            embedding = embedding_model.encode([text]).astype(
                "float32"
            )  # Ensure float32 for FAISS
            index.add(embedding)
            retrieval_knowledge_store.append(text)
            retrieval_knowledge_set.add(text)  # {{ Update the set }}
            logging.info(
                "Retrieval knowledge added to FAISS index and knowledge store."
            )
            return embedding
    except Exception as e:
        logging.error(f"Error adding retrieval knowledge: {e}")


def add_training_knowledge(interaction, training_knowledge_store):
    logging.debug(f"Adding training knowledge: {interaction}")
    try:
        training_knowledge_store.append(interaction)
        logging.info("Interaction added to training knowledge store.")
    except Exception as e:
        logging.error(f"Error adding training knowledge: {e}")


def save_retrieval_faiss_index(index, retrieval_faiss_index_file):
    logging.debug(f"Saving FAISS index to {retrieval_faiss_index_file}")
    try:
        faiss.write_index(index, retrieval_faiss_index_file)
        logging.info("Retrieval FAISS index saved.")
    except Exception as e:
        logging.error(f"Error saving FAISS index: {e}")


def save_training_knowledge_store(
    training_knowledge_store, training_knowledge_store_file
):
    logging.debug(f"Saving training knowledge store to {training_knowledge_store_file}")
    try:
        with open(training_knowledge_store_file, "w") as f:
            json.dump(training_knowledge_store, f)
        logging.info("Training knowledge store saved.")
    except Exception as e:
        logging.error(f"Error saving training knowledge store: {e}")


def save_all(
    index,
    retrieval_faiss_index_file,
    training_knowledge_store,
    training_knowledge_store_file,
    retrieval_knowledge_store,
    retrieval_knowledge_set,
):
    logging.debug("Saving all knowledge stores.")
    save_retrieval_faiss_index(index, retrieval_faiss_index_file)
    save_training_knowledge_store(
        training_knowledge_store, training_knowledge_store_file
    )
    # Save retrieval knowledge store
    logging.debug(
        f"Saving retrieval knowledge store to {RETRIEVAL_KNOWLEDGE_STORE_FILE}"
    )
    try:
        with open(RETRIEVAL_KNOWLEDGE_STORE_FILE, "w") as f:
            json.dump(retrieval_knowledge_store, f)
        logging.info("Retrieval knowledge store saved.")
    except Exception as e:
        logging.error(f"Error saving retrieval knowledge store: {e}")
    # Note: The set does not need to be saved as it can be reconstructed on load


def get_recent_knowledge(retrieval_knowledge_store, limit=5):
    logging.debug(f"Retrieving the {limit} most recent knowledge entries.")
    try:
        return retrieval_knowledge_store[-limit:]
    except Exception as e:
        logging.error(f"Error retrieving recent knowledge: {e}")
        return []


def get_relevant_knowledge(
    query, embedding_model, index, retrieval_knowledge_store, top_k=5
):
    logging.debug(f"Getting relevant knowledge for query: {query}")
    try:
        # Get the most relevant knowledge from the FAISS index
        query_embedding = embedding_model.encode([query]).astype("float32")
        distances, indices = index.search(query_embedding, top_k)
        relevant_knowledge = [
            retrieval_knowledge_store[idx]
            for idx in indices[0]
            if idx < len(retrieval_knowledge_store)
        ]
        logging.debug(f"Relevant knowledge retrieved: {relevant_knowledge}")
        return relevant_knowledge
    except Exception as e:
        logging.error(f"Error getting relevant knowledge: {e}")
        return []


def extract_knowledge(conversation_history, preferred_response):
    logging.debug("Extracting knowledge from interactions.")
    try:
        SEMANTIC_PROMPT = load_prompt("prompts/semantic_knowledge.txt")
        EPISODIC_PROMPT = load_prompt("prompts/episodic_knowledge.txt")
        if preferred_response:
            conversation_history.append(
                {
                    "role": "user",
                    "content": "[PREFERRED RESPONSE]\n" + preferred_response,
                }
            )
            logging.debug(f"Preferred response appended: {preferred_response}")
        # Maintain conversation_history as a single string
        conversation_content = "\n".join(
            [msg["content"] for msg in conversation_history]
        )
        try:
            messages = [
                {"role": "system", "content": SEMANTIC_PROMPT},
                {"role": "user", "content": conversation_content},
            ]
            outputs = pipe(
                messages,
                max_new_tokens=8192,  # {{ Reduced for efficiency }}
            )
            semantic_response = outputs[0]["generated_text"][-1]
            logging.debug(f"Generated semantic_response: {semantic_response}")
        except Exception as e:
            logging.error(f"Error generating semantic response: {e}")
            semantic_response = (
                "I'm sorry, I encountered an error while generating semantic knowledge."
            )

        try:
            messages = [
                {"role": "system", "content": EPISODIC_PROMPT},
                {"role": "user", "content": conversation_content},
            ]
            outputs = pipe(
                messages,
                max_new_tokens=8192,  # {{ Reduced for efficiency }}
            )
            episodic_response = outputs[0]["generated_text"][-1]
            logging.debug(f"Generated episodic_response: {episodic_response}")
        except Exception as e:
            logging.error(f"Error generating episodic response: {e}")
            episodic_response = (
                "I'm sorry, I encountered an error while generating episodic knowledge."
            )

        logging.debug("Knowledge extraction completed successfully.")
        return semantic_response, episodic_response
    except Exception as e:
        logging.error(f"Error extracting knowledge: {e}")
        return "", ""


def get_training_knowledge_count(training_knowledge_store):
    """
    Returns the number of training knowledge examples.
    """
    try:
        return len(training_knowledge_store)
    except Exception as e:
        logging.error(f"Error getting training knowledge count: {e}")
        return 0


def load_retrieval_knowledge_store(file_path):
    logging.debug(f"Loading retrieval knowledge store from {file_path}")
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
        retrieval_knowledge_set = set(data)
        logging.info(
            "Retrieval knowledge store loaded with set for duplication checks."
        )
        return data, retrieval_knowledge_set  # {{ Return both list and set }}
    except Exception as e:
        logging.error(f"Error loading retrieval knowledge store: {e}")
        return [], set()
