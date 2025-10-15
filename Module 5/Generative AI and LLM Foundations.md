### **Generative AI and LLM Foundations**

**Introduction to Generative AI** 

**1\. Overview**

* **AI**: Enables machines to imitate human intelligence.

* **Machine Learning (ML)**: Subset of AI; algorithms learn from past data to make predictions.

* **Deep Learning (DL)**: Subset of ML using neural networks to learn complex data.

* **Generative AI (GenAI)**: Subset of DL that **creates new content** (text, images, audio, video, code).

**2\. How Generative AI Works**

* Learns **patterns** from large datasets.

* Uses those patterns to **generate new data** with similar characteristics.

* Example: Learning from thousands of dog images → generating a new image of a dog (not a copy).

* Involves **complex math and neural network architectures**.

**3\. Difference Between ML and GenAI**

| Aspect | Machine Learning | Generative AI |
| ----- | ----- | ----- |
| **Input Data** | Labeled (features \+ labels) | Mostly unlabeled (unstructured) |
| **Learning Goal** | Learn relation between data & labels | Learn patterns in unstructured data |
| **Output** | Prediction / Classification | New data (text, image, audio, etc.) |
| **Use Case** | Identify or predict | Create or generate |

**4\. Training Process**

* **ML:** Supervised learning → model trained on labeled data (e.g., “cat” or “dog”).

* **GenAI:** Pre-trained on **unlabeled** data; later can be **fine-tuned** with labeled data for specific tasks.

**5\. Types of Generative AI Models**

* **Text-Based Models:** Generate text, code, articles, chat responses (e.g., GPT models).

* **Multimodal Models:** Handle multiple data types — text, images, audio, video simultaneously (e.g., image captioning, video generation).

  ---

**6\. Applications of Generative AI**

* **Content creation** (text, images, videos, music).

* **Chatbots & virtual assistants.**

* **Medical imaging & drug discovery.**

* **Design, gaming, and automation.**

* Drives creativity and innovation across industries.

- **Introduction to Large Language Models**

**Large Language Models (LLMs) — Short Notes**

**1\. Definition**

* A **language model** is a **probabilistic model of text** that predicts the likelihood of a word appearing next in a sequence.

* It computes a **probability distribution** over a fixed **vocabulary** based on context.

* Example: “I wrote to the zoo to send me a pet. They sent me a \_\_\_.”

  * Model assigns probabilities: *dog (0.45)*, *lion (0.03)*, *elephant (0.02)* → chooses “dog”.

**2\. Large Language Models (LLMs)**

* “Large” refers to the **number of parameters** (no fixed threshold; can range from millions to billions).

* **Parameters** are adjustable weights in neural networks optimized during training.

* **Model size** \= memory required to store parameters and data structures.

**3\. Working Process**

1. Model predicts probabilities for all possible next words.

2. Word with the **highest probability** is selected and appended.

3. Probabilities update dynamically for the next prediction.

4. Process continues until an **EOS (End of Sentence)** token is generated.

**4\. Capabilities of LLMs**

* **Question Answering** – factual and reasoning-based queries.

* **Text Generation** – articles, essays, summaries.

* **Translation** – e.g., English to French.

* **Sentiment Analysis, Summarization, Reasoning**, etc.

**5\. Architecture — Transformer**

* LLMs use **Transformer architecture**.

* Uses **attention mechanisms** to focus on relevant parts of input for next-word prediction.

* Provides **contextual understanding** and long-range dependency handling.

**6\. Data & Training**

* Trained on **massive text datasets** from the internet.

* Part of **Deep Learning** → subset of Machine Learning → subset of AI.

* Models learn linguistic structure, semantics, and world knowledge.

**7\. Growth and Limitations**

* Parameter counts have grown exponentially (hundreds of millions → hundreds of billions).

* Larger models often perform better but risk **overfitting** if too large.

* More parameters ≠ always better performance.

**8\. Summary**

* LLMs are **deep learning-based probabilistic models** that predict and generate human-like text.

* Built using **Transformer architecture**, trained on large-scale data.

* Core to **Generative AI** applications like ChatGPT, translation, and summarization.

- **Transformer Architecture — Part 1** 

**1\. Challenge in Language Understanding**

* Sentence example: *“Jane threw the frisbee and her dog fetched it.”*

* Humans easily know *“it”* refers to *“frisbee”*, but machines struggle to capture such context.

**2\. RNN (Recurrent Neural Networks)**

* Designed for **sequential data** (text, speech).

* Have a **feedback loop** allowing information to persist through **hidden states**.

* Process input **one element at a time** → capture short-term dependencies well.

* **Limitation:** Struggles with **long-range dependencies** → *vanishing gradient problem*.

* As sequence length grows, context fades → model forgets earlier words.

**3\. Transformer Architecture — Overview**

* **Key Idea:** Understand relationships between **all words simultaneously**.

* Provides a **bird’s-eye view** of the sentence → captures **global context**.

* Can link distant words (e.g., *Jane* ↔ *dog*) in a single pass.

* Overcomes RNN limitations by processing input **in parallel**, not sequentially.

**4\. Self-Attention Mechanism (Attention Mechanism)**

* Core feature of transformers.

* Allows model to **weigh importance** of each word relative to others.

* Example: In *“Jane threw the frisbee and her dog fetched it”*, model learns *“it”* refers to *“frisbee”*.

* Captures **contextual and long-range dependencies** effectively.

**5\. Transformer Paper**

* Introduced in *“Attention Is All You Need” (Vaswani et al., 2017).*

* Replaced RNNs with **attention-based architecture**.

**6\. Encoder–Decoder Structure**

* **Encoder:** Processes input text → converts to **numerical representations (vectors)** capturing meaning.

* **Decoder:** Uses encoded vectors → **generates output text**.

* Both encoder and decoder contain **multiple layers**, connected by **self-attention**.

**7\. Key Takeaways**

* Transformers understand **context holistically**, not word-by-word.

* **Self-attention** enables contextual weighting and long-distance understanding.

* Foundation for modern **large language models** (LLMs).

- **Transformer Architecture — Part 2**

**1\. Encoder and Decoder Overview**

* **Encoder:** Reads input text → converts it into **embeddings** (numerical form capturing meaning).

* **Decoder:** Uses these embeddings to generate **output text**.

* Together, they form the **encoder–decoder architecture** used in many transformer models.

**2\. Tokens and Tokenization**

* Models process **tokens**, not words.

* A **token** can be:

  * A complete word (*apple*)

  * A part of a word (*friend \+ ship*)

  * A punctuation mark (*comma, colon, period*)

* Average:

  * **Simple text:** \~1 token per word

  * **Complex text:** \~2–3 tokens per word

**3\. Embeddings**

* **Embeddings \= Numerical (vector) representations** of text (word, phrase, or sentence).

* Help machines understand **semantic relationships** between text fragments.

* **Encoder model** converts tokens → embeddings.

* Applications:

  * **Classification tasks**

  * **Vector databases**

  * **Semantic search** → finding similar documents by meaning, not keywords.

**4\. Retrieval-Augmented Generation (RAG)**

* Combines **retrieval \+ generation**.

* Steps:

  1. Encode all documents → store embeddings in **vector database**.

  2. Encode user query → find **most similar embeddings**.

  3. Retrieve relevant content → send to **LLM** for an informed answer.

* **Embeddings quality** is critical for accurate retrieval.

**5\. Decoder Models**

* Generate next token **one at a time** based on previous tokens.

* Predicts next token using **probability distribution** over vocabulary.

* Can be invoked repeatedly to produce full text.

* Used for **text generation** (articles, stories, etc.).

**6\. Encoder–Decoder Models**

* Combine encoder and decoder for **sequence-to-sequence tasks** (e.g., translation).

* Workflow:

  1. Input sentence → encoder → embeddings

  2. Decoder → generates tokens **sequentially**

  3. Each new token is fed back into the decoder to predict the next one

**7\. Summary of Transformer Model Types**

| Model Type | Function | Use Case |
| ----- | ----- | ----- |
| **Encoder-only** | Converts input to vector embeddings | Semantic search, RAG, classification |
| **Decoder-only** | Generates output sequences | Text generation, content creation |
| **Encoder–Decoder** | Maps input sequence to output sequence | Translation, summarization |

**Key Takeaway:**

Transformers can be **encoder-based**, **decoder-based**, or **hybrid** depending on task — from understanding data (semantic search) to generating new text (LLMs like GPT).

- **Prompt Engineering** 

**1\. Definition**

* **Prompt:** Input text given to a Large Language Model (LLM).

* **Prompt Engineering:** Iteratively refining prompts to get desired, accurate, and contextual responses.

* LLMs predict the **next word/token** in a sequence — hence, prompts guide their output style and intent.

**2\. From Completion Models to Instruction-Tuned Models**

* **Completion LLMs:** Trained to *continue text* (not to follow instructions).

* **Limitation:** Hard to control output or perform specific tasks.

* **Instruction Tuning:** Fine-tunes models using instruction–response pairs to make them follow human intent.

  * Example: **Llama 2 Chat** fine-tuned on \~28,000 prompt–response pairs.

**3\. Reinforcement Learning from Human Feedback (RLHF)**

* Fine-tunes LLMs using **human feedback** to align responses with human preferences.

* **Process:**

  1. Human annotators create prompts and rate model outputs.

  2. A **reward model** learns these preferences.

  3. Model is trained to optimize for human-aligned outputs.

* Most modern chat-based LLMs (like ChatGPT) use **instruction tuning \+ RLHF**.

**4\. Prompting Strategies**

**(a) In-Context Learning & k-Shot Prompting**

* Model is given examples or context *within* the prompt.

* **0-shot:** No examples

* **1-shot:** One example

* **k-shot:** k examples

* Example: “Translate English to French” → give 3 examples → ask model to translate *cheese*.

* **Few-shot prompting** generally improves accuracy over 0-shot.

**(b) Chain-of-Thought (CoT) Prompting**

* Encourages the model to **show reasoning steps** before the final answer.

* Breaks complex problems into smaller reasoning steps.

* Example:

  * Question: “Roger has 5 tennis balls, buys 2 cans (3 balls each). Total?”

  * Response: “5 \+ (2×3) \= 11\. Answer: 11.”

* Leads to more accurate and interpretable answers.

**5\. Hallucination in LLMs**

* **Definition:** Model-generated text that is factually incorrect or ungrounded in data.

* **Example:** “Americans drive on the left side of the road.” → False statement \= hallucination.

* **Issue:** Hallucinations can be **fluent but misleading** and sometimes subtle.

* **Reduction methods:**

  * **Retrieval-Augmented Generation (RAG)** → reduces hallucination by grounding outputs in real data.

  * No complete solution yet; research continues on **measuring groundedness**.

**6\. Key Takeaways**

* Prompts define **what** and **how** the model responds.

* **Instruction tuning \+ RLHF** make LLMs task-oriented.

* **Few-shot** and **chain-of-thought** prompting are proven to improve performance.

* **Hallucination** remains a core challenge — retrieval-based systems help but don’t eliminate it.

- **Customizing LLMs with Your Own Data** 

### **1\. Framework Overview**

* Two dimensions of customization:

  1. **Context Optimization (Horizontal):** Provide *specific data or context* (e.g., user info, order history).

  2. **LLM Optimization (Vertical):** Adapt the model to a *specific task or domain* (e.g., legal, finance).

* Three key approaches:

  1. **Prompt Engineering** – Fastest, simplest customization.

  2. **Retrieval-Augmented Generation (RAG)** – Adds external data context.

  3. **Fine-Tuning** – Adapts model weights for specialized performance.

* These methods are **additive** and can be combined.

### **2\. Retrieval-Augmented Generation (RAG)**

* **Purpose:** Ground model outputs in factual, enterprise data without retraining.

* **Components:**

  * **Retrieval:** Search relevant data (e.g., knowledge base, vector DB).

  * **Augmented Generation:** Use retrieved data to form a factual, grounded response.

* **Example:**

  * Chatbot retrieves company return policy from vector DB before answering customer.

* **Advantages:**

  * No fine-tuning needed.

  * Reduces hallucination; uses latest enterprise data.

* **Use Case:** Accessing private or dynamic information (e.g., support, policies).

* **Limitation:** Requires good data quality and setup.

### **3\. Fine-Tuning**

* **Definition:** Train a pre-trained model on *domain-specific or task-specific data*.

* **Steps:**

  * Start with a base LLM.

  * Add custom, labeled data.

  * Train → results in a **fine-tuned model** specialized in domain language and tone.

* **OCI Example:** **T-Few fine-tuning** – updates only select layers to reduce cost/time.

* **Benefits:**

  * Improves accuracy for specialized tasks.

  * Increases efficiency (smaller, faster model).

* **Limitations:**

  * Requires labeled dataset.

  * High cost and time for training.

  * More resource-intensive.

### **4\. When to Use Which**

| Method | When to Use | Advantages | Limitations |
| ----- | ----- | ----- | ----- |
| **Prompt Engineering** | Model already knows domain concepts. | Simple, fast, no cost. | Limited control, may not handle new data. |
| **RAG** | Data changes frequently or needs grounding. | Uses current data, reduces hallucination. | Needs quality database, complex setup. |
| **Fine-Tuning** | Model performs poorly or domain-specific task. | Custom, efficient model. | Costly, needs labeled data and compute. |

### **5\. Combined Usage Framework**

* **Step 1:** Start with **Prompt Engineering** → test performance.

* **Step 2:** Add **Few-shot examples** to improve accuracy.

* **Step 3:** If external knowledge is needed → integrate **RAG**.

* **Step 4:** If output style or domain knowledge still lacking → apply **Fine-Tuning**.

* **Step 5:** Iterate to optimize both **retrieval** and **model performance**.

### **6\. Key Takeaways**

* **Prompt Engineering:** Quick baseline improvement.

* **RAG:** Adds factual grounding and domain access.

* **Fine-Tuning:** Deep domain adaptation and efficiency.

* Combining all three provides **maximum accuracy, adaptability, and reliability** for enterprise LLM applications.

