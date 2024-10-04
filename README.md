# Continuous Learning Assistant

## Overview

The Continuous Learning Assistant is an intelligent chat-based system designed to interact with users, process their inputs, retrieve relevant knowledge, generate meaningful responses, collect feedback, and continuously improve its performance through ongoing training. Leveraging advanced technologies such as transformer models, FAISS for efficient knowledge retrieval, and Direct Preference Optimization (DPO) for fine-tuning, this assistant aims to provide accurate and contextually aware assistance.

## Features

- **User Interface**: Interactive chat interface for seamless user interaction.
- **Processing Pipeline**: Handles preprocessing, context management, and Named Entity Recognition (NER) extraction.
- **Response Generation**: Utilizes GPT models enhanced with Low-Rank Adaptation (LoRA) for generating responses.
- **Knowledge Retrieval**: Employs FAISS for efficient retrieval from a comprehensive knowledge store.
- **Feedback Collection**: Gathers explicit and implicit user feedback to refine responses.
- **Continuous Training**: Asynchronously fine-tunes the model based on accumulated interactions and feedback using DPO.

## Architecture

![Workflow Diagram](diagram.md)

1. **User Interface**: The front-end chat interface where users interact with the assistant.
2. **Processing Pipeline**: Preprocesses user inputs, manages context, and extracts named entities.
3. **Response Generation**: Generates responses using a GPT model integrated with LoRA for adaptability.
4. **Knowledge Retrieval**: Retrieves relevant information from the FAISS-indexed knowledge store to inform responses.
5. **Feedback Loop**: Collects user feedback to assess and improve response quality.
6. **Training Pipeline**: Periodically fine-tunes the model using collected data to enhance performance.

## Installation

### Prerequisites

- Python 3.8+
- CUDA-enabled GPU (optional, for accelerated processing)
- [FAISS](https://github.com/facebookresearch/faiss) installed

### Steps

1. **Clone the Repository**
   ```bash
   git clone https://github.com/peytontolbert/continuous-learning-assistant.git
   cd continuous-learning-assistant
   ```

2. **Create a Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Models and Indexes**
   - Ensure that the necessary transformer models are downloaded.
   - Initialize or place the FAISS index files in the project directory.

## Usage

Run the main application:

```
bash
python main.py
```

Interact with the assistant through the command-line interface. Type your messages and receive responses in real-time. To exit the application, type `exit` or `quit`.

## Configuration

All configurations are managed within `main.py`, including:

- **Embedding Model**: Defined by `EMBEDDING_MODEL_NAME`.
- **FAISS Index Files**: Paths specified for retrieval and training knowledge stores.
- **Training Parameters**: Includes training intervals and minimum sample requirements.
- **Model Directories**: Specifies where fine-tuned models are saved.

## File Structure

- `main.py`: Entry point of the application handling user interactions and orchestrating components.
- `models/transformer.py`: Handles response generation using transformer models.
- `training/fine_tune_dpo.py`: Manages the fine-tuning process using DPO based on user interactions.
- `util_scripts/knowledge.py`: Functions for managing knowledge stores and retrieval processes.
- `prompts/`: Contains prompt templates for semantic and episodic knowledge extraction.
- `workflow.md`: Visual representation of the system's workflow.
- `.gitignore`: Specifies files and directories to be ignored by Git.

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a new feature branch.
3. Commit your changes with clear messages.
4. Push to your forked repository.
5. Submit a pull request detailing your enhancements.

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgements

- [Transformers by Hugging Face](https://github.com/huggingface/transformers)
- [FAISS by Facebook AI](https://github.com/facebookresearch/faiss)
- [Sentence Transformers](https://www.sbert.net/)