+--------------------+          +-------------------------------+          +----------------------+
|                    |          |                               |          |                      |
|   User Interface   +--------->+      Processing Pipeline      +--------->+  Response Generation |
| (Chat Interface)   |          | (Preprocessing, Context       |          |  (GPT Model + LoRA)  |
|                    |          |  Management, NER Extraction)  |          |                      |
+--------------------+          +-------------------------------+          +----------------------+
                                           |                                       |
                                           v                                       |
                              +-------------------------+                          |
                              |  Retrieval Knowledge    |                          |
                              |  Store (FAISS Index)    |                          |
                              +-------------------------+                          |
                                           |                                       |
                                           v                                       |
                              +-----------------------------+                      |
                              | Generate Response for user  |                      |
                              +-----------------------------+                      |
                                           |                                       |
                                           v                                       v
                              +-------------------------+          +-------------------------------+
                              |  Feedback Collection    +<---------+   Training Knowledge Store    |
                              |  (Explicit/Implicit)    |          |  (User Inputs, Responses,     |
                              +-------------------------+          |   Feedback, Metadata)         |
                                           |                       +-------------------------------+
                                           v
                              +-------------------------+
                              |  Asynchronous Training  |
                              |     Pipeline            |
                              |  (Fine-Tuning with      |
                              |   Backpropagation)      | 
                              +-------------------------+

- Assistant needs both knowledge store and training pipeline.
- Training pipeline is to provide better prediction on responses. 
- Training pipeline utilizes knowledge store and a pretrained model to generate a DPO example from a user query and an assistant response. 
- The DPO example is then used to fine-tune the model using PEFT.
- The knowledge store is used to store the knowledge that the assistant has learned.
- The knowledge store is updated with new knowledge when the assistant learns something new.
- The knowledge store is compressed using FAISS to store the knowledge in a more efficient way.
