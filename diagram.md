```mermaid
graph TD
    UI[User Interface(Chat Interface)] --> PP[Processing Pipeline\n(Preprocessing, Context Management, NER Extraction)]
    PP --> RG[Response Generation\n(GPT Model + LoRA)]
    PP --> RK[Retrieval Knowledge Store\n(FAISS Index)]
    RG --> FBC[Feedback Collection\n(Explicit/Implicit)]
    RG --> TKS[Training Knowledge Store\n(User Inputs, Responses, Feedback, Metadata)]
    FBC --> ATP[Asynchronous Training Pipeline\n(Fine-Tuning with Backpropagation)]
    RK --> ATP
```

- Assistant needs both knowledge store and training pipeline.
- Training pipeline is to provide better prediction on responses. 
- Training pipeline utilizes knowledge store and a pretrained model to generate a DPO example from a user query and an assistant response. 
- The DPO example is then used to fine-tune the model using PEFT.
- The knowledge store is used to store the knowledge that the assistant has learned.
- The knowledge store is updated with new knowledge when the assistant learns something new.
- The knowledge store is compressed using FAISS to store the knowledge in a more efficient way.
