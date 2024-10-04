```mermaid
graph TD
    UI[User Interface<br>(Chat Interface)] --> PP[Processing Pipeline<br>(Preprocessing,<br>Context Management,<br>NER Extraction)]
    PP --> RG[Response Generation<br>(GPT Model + LoRA)]
    
    PP --> RK[Retrieval Knowledge Store<br>(FAISS Index)]
    RK --> GR[Generate Response for User]
    
    GR --> FC[Feedback Collection<br>(Explicit/Implicit)]
    GR --> TKS[Training Knowledge Store<br>(User Inputs, Responses,<br>Feedback, Metadata)]
    
    FC --> TKS
    
    TKS --> ATP[Asynchronous Training Pipeline<br>(Fine-Tuning with Backpropagation)]
    ATP --> RG
```