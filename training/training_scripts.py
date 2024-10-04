import torch
from torch.utils.data import DataLoader, Dataset
import json
from models.griffin import GriffinModel  # Ensure correct import path
from torch import nn
import torch.optim as optim
from transformers import AutoTokenizer


# Define a simple TextDataset if not already defined
class TextDataset(Dataset):
    def __init__(self, data, seq_len, stoi):
        self.data = data
        self.seq_len = seq_len
        self.stoi = stoi

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        encoded = encode(self.data[idx], self.stoi)
        input_seq = encoded[: self.seq_len]
        target_seq = encoded[1 : self.seq_len + 1]
        return torch.tensor(input_seq, dtype=torch.long), torch.tensor(
            target_seq, dtype=torch.long
        )


def encode(s, stoi):
    return [stoi.get(c, 0) for c in s]  # Use 0 for unknown characters


def train_griffin_model(user_input, knowledge_text, training_target):
    # Configuration
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    learning_rate = 1e-3
    batch_size = 4
    num_epochs = 1  # Adjust epochs as needed for real-time
    input_dim = 768
    rnn_width = 1024
    depth = 12
    mlp_expansion_factor = 3
    SEQ_LEN = 256
    TRAINING_KNOWLEDGE_STORE_FILE = "griffin_training_knowledge_store.json"

    # Load existing training data
    try:
        with open(TRAINING_KNOWLEDGE_STORE_FILE, "r") as f:
            training_data = json.load(f)
    except FileNotFoundError:
        training_data = []

    # Append new training example
    training_data.append(
        {
            "user_input": user_input,
            "knowledge_text": knowledge_text,
            "training_target": training_target,
        }
    )

    # Save updated training data
    with open(TRAINING_KNOWLEDGE_STORE_FILE, "w") as f:
        json.dump(training_data, f)

    # Initialize the pretrained tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "bert-base-uncased"
    )  # Replace with your desired tokenizer

    # Prepare dataset
    pairlist = [
        item["user_input"] + tokenizer.sep_token + item["training_target"]
        for item in training_data
    ]
    vocab_size = tokenizer.vocab_size

    encoded_data = [
        tokenizer.encode(pair, add_special_tokens=True) for pair in pairlist
    ]
    train_dataset = TextDataset(encoded_data, SEQ_LEN, tokenizer)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Initialize or load the model
    model = GriffinModel(vocab_size, input_dim, mlp_expansion_factor, rnn_width, depth)
    model.to(device)

    try:
        model.load_state_dict(torch.load("fine_tuned_griffin_model.pth"))
        model.train()
    except FileNotFoundError:
        print("No existing model found. Initializing a new model.")
        model.train()

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(num_epochs):
        for batch_idx, (inputs, targets) in enumerate(train_dataloader):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs.view(-1, vocab_size), targets.view(-1))
            loss.backward()
            optimizer.step()

            if batch_idx % 10 == 0:
                print(f"Batch {batch_idx}, Loss: {loss.item()}")

    # Save the fine-tuned model
    torch.save(model.state_dict(), "fine_tuned_griffin_model.pth")
    print("Model training completed and saved.")
