### **OCI Generative AI Service**

## **OCI Generative AI** 

### **What it is**

* Fully managed **generative AI service** on OCI.

* Provides **customizable large language models (LLMs)** via **single API access**.

* **Serverless**: No infrastructure management required.

* Key use: Build generative AI applications like chat, text generation, semantic search, and information retrieval.

  ### **Key Characteristics**

1. **Pre-trained Foundational Models**

   * Ready-to-use large language models from **Meta** and **Cohere**.

   * Two types:

     * **Chat models**: Maintain context and handle conversations.

       * **Command-R-Plus**: Handles larger prompts (up to 128k tokens), higher capacity, more expensive.

       * **Command-R**: Handles smaller prompts (16k tokens), entry-level, more affordable.

       * **LLaMA 3 70B Instruct**: Instruction-following model for human instructions (summarization, email generation, etc.).

     * **Embedding models**: Convert text into numerical vectors for semantic understanding.

       * **Embed English** and **Embed Multilingual**.

       * Use cases: Semantic search, cross-language search.

       * Multilingual model supports **100+ languages**.

2. **Flexible Fine-Tuning**

   * Customize pre-trained models on **domain-specific datasets**.

   * Benefits:

     * Improves performance on **specific tasks or domains**.

     * Increases model efficiency.

   * **T-Few Fine-Tuning**:

     * Inserts new layers and updates only a fraction of model weights.

     * Reduces **training time** and **cost** compared to full fine-tuning.

3. **Dedicated AI Clusters**

   * **GPU-based compute resources** for fine-tuning and inference workloads.

   * Key features:

     * **Dedicated GPU pool** for customer workloads.

     * **Exclusive RDMA cluster networking** for ultra-low latency.

     * Security: GPUs are isolated per customer task.

   * Supports scaling to large GPU clusters efficiently.

   ### **How it Works**

1. Provide **text input** (prompt) to the model.

2. Optional: Submit additional **contextual text** (documents, emails, reviews).

3. Service **processes the input** and generates intelligent outputs.

   ### **Use Cases**

* **Chatbots**: Dialogue-based interactions.

* **Text Generation**: Emails, summaries, content creation.

* **Semantic Search**: Meaning-based search rather than keyword-based.

* **Information Retrieval**: Querying large datasets intelligently.

  ### **Summary**

* **OCI Generative AI Service** is designed for **flexibility, scalability, and security**:

  1. Pre-trained foundational models for quick deployment.

  2. Fine-tuning for domain-specific tasks.

  3. Dedicated AI clusters for efficient, secure GPU compute.

  ## **Demo: OCI Generative AI Service** 

  ## **Getting Started**

1. **Login to OCI Console**

   * Make sure the service is available in your region.

   * Current demo: **Germany Central (Frankfurt)**.

2. **Accessing Generative AI Service**

   * Navigate: **Burger Menu → Analytics & AI → AI Services → Generative AI**.

   * Dashboard shows:

     * **Playground** – Visual interface to test models without code.

     * **Dedicated AI Clusters** – GPU compute resources for fine-tuning and hosting.

     * **Custom Models** – Fine-tuned models.

     * **Endpoints** – For hosting models for inference.

   ### **Playground**

1. **Pre-trained Models**

   * **Chat Models**:

     * **Command-R-Plus**: Up to 128,000 tokens, high capacity.

     * **Command-R**: Up to 16,000 tokens, affordable.

     * **Meta LLaMA 3 70B Instruct**: Up to 8,000 tokens.

   * **Embedding Models**:

     * **Embed English** and **Embed Multilingual**.

     * Convert text to numerical vectors for **semantic search**.

     * Multilingual model supports 100+ languages.

2. **Using Chat Models**

   * Maintain context across multiple prompts.

   * Can ask follow-up questions.

   * Options to:

     * **View Code**: Python/Java client code for API integration.

     * **Preamble Override**: Adjust model behavior (style, tone, persona).

     * **Temperature**: Control randomness of output.

3. **Using Embedding Models**

   * Convert text to vectors for **semantic similarity**.

   * Example: HR Help Center articles clustered by topics (skills, vacation, etc.).

   * Visualizes similarity in **2D**; in reality, embeddings have many dimensions.

   * Useful for **semantic search** and information retrieval.

   ### **Dedicated AI Clusters**

* **GPU-based compute resources** for fine-tuning and inference.

* Steps to create:

  1. Click **Create Dedicated AI Cluster**.

  2. Choose **cluster purpose**: hosting or fine-tuning.

  3. Select **pre-trained model**.

  4. Click **Create**.

  ### **Fine-Tuning Models**

* Steps to create a custom fine-tuned model:

  1. Click **Create Model**.

  2. Provide model **name**.

  3. Select **base pre-trained model**.

  4. Choose **fine-tuning method**.

  5. Ensure **dedicated AI cluster** is available for compute.

  ### **Creating Endpoints**

* Steps to serve inference traffic:

  1. Click **Create Endpoint**.

  2. Provide **name** and **hosting configuration**.

  3. Select the **fine-tuned model** to host.

  4. Assign a **dedicated AI cluster**.

     **Key Takeaways**  
* **Playground** is ideal for testing prompts and exploring model behavior without writing code.

* **Code generation** allows seamless API integration into Python/Java applications.

* **Fine-tuning** allows domain-specific customization.

* **Dedicated AI clusters** provide secure, isolated GPU resources for large-scale inference and training.

* Embedding models enable **semantic understanding and search**.

* Chat models retain **context**, enabling multi-turn conversations.

## **Oracle AI Vector Search**

### **Introduction**

* Oracle Database 23ai has **built-in AI Vector Search**.

* Supports **converged database approach**: relational, JSON, XML, graph, spatial, text, and vector data.

* AI Vector Search enables **semantic-based similarity searches** across structured and unstructured data.

* Powers **Gen AI pipelines** with enterprise data.

### **Core Components**

1. **VECTOR Datatype**

   * Stores embeddings generated by models.

   * Can be used with relational data or standalone.

   * Flexible schema supports evolving embedding models.

2. **VECTOR\_EMBEDDING Function**

   * Generates vector embeddings from models (API call or ONNX model loaded into DB).

   * Example: `resnet_50` for image embeddings.

3. **VECTOR\_DISTANCE Function**

   * Computes similarity/distance between vectors.

   * Smaller distance → more similar entities.

   * Supports different **distance metrics** (default: cosine).

### **Vector Search in SQL**

* Find closest matches using SQL queries with **VECTOR\_DISTANCE**.

* Example: matching top 10 job positions to a candidate’s resume.

* Fully integrated in a single query combining:

  * Applicant data

  * Job data

  * AI Vector Search

### **Vector Index**

* Improves **performance** and controls **accuracy** of similarity searches.

* Created like standard table indexes.

* Options:

  * **Organization**: INMEMORY\_NEIGHBOR\_GRAPH (fits in memory) or NEIGHBOR\_PARTITIONS.

  * **DISTANCE**: specify metric (cosine, etc.).

  * **TARGET ACCURACY**: controls approximate search quality.

* Approximate searches:

  * Example: `FETCH TOP 5 APPROXIMATE` returns 4/5 matches with target accuracy 80%.

### **Joins and Enterprise Data**

* Supports **similarity searches over joins**.

* Example: querying `Author`, `Books`, and `Pages` tables simultaneously.

* Each page has its own vector embedding.

* Cost-based optimizer decides:

  * Join plan

  * Vector index usage for performance.

### **Gen AI Pipeline Integration**

* Data sources: database tables, CSV files, social media, etc.

* Workflow:

  1. Load documents (Document Loader)

  2. Transform documents (split, summarize)

  3. Generate embeddings

  4. Store in vector columns

  5. Perform similarity searches or RAG (retrieval-augmented generation) with LLMs

* Integration with **LangChain** and **LamaIndex** for advanced applications.

### **Key Advantages**

* Fully integrated **vector search within Oracle Database**.

* Supports **enterprise-grade performance, reliability, and security**.

* Combines relational, document, spatial, graph, machine learning, and AI vector queries in **single SQL statements**.

* Efficient orchestration of Gen AI pipelines, either **natively** or with **third-party frameworks**.

### **Summary**

* Oracle Database 23ai enables **semantic search at scale**.

* Provides **VECTOR datatype**, **VECTOR\_EMBEDDING**, **VECTOR\_DISTANCE**, and **vector indexing**.

* Can power **advanced Gen AI applications** and pipelines directly inside the database.

## **Oracle Select AI**

## **Introduction**

* **Select AI** enables **natural language queries** on Oracle Autonomous Database.

* No need to know table names, columns, or SQL syntax.

* Leverages **large language models (LLMs)** to interpret queries and generate SQL.

* Part of Oracle’s **AI ecosystem**, integrating with OCI Gen AI services.

### **Core Features**

1. **Natural Language Queries**

   * Ask questions in plain English (or other supported languages).

   * Example: “Top 10 streamed movies.”

   * Can follow up with related questions, and the system keeps context.

2. **Automatic SQL Generation**

   * LLM interprets intent and database structure.

   * Generates SQL statements automatically.

   * SQL can be reviewed using `SELECT AI SHOWSQL`.

3. **Integration with Applications**

   * Works with **APEX** and other tools to visualize data.

   * Supports charts, interactive reports, and filtered views.

   * Can be embedded in custom apps.

### **How It Works**

1. **User Query**

   * User asks a question in natural language.

2. **AI Profile**

   * Encapsulates model selection, schema, tables, and views.

   * Models can come from:

     * **OCI Generative AI**

     * **Cohere**

     * **OpenAI**

     * **LLaMA**

3. **Prompt Generation**

   * Database generates a prompt for the LLM based on metadata.

4. **SQL Generation**

   * LLM produces SQL query that runs against the database.

5. **Result Delivery**

   * Returns:

     * Result set

     * Narrative explanation

     * SQL for review or further use

### **SQL Examples**

* `SELECT AI "Top customer segments for George Clooney movies";`

* `SELECT AI SHOWSQL ...` → view the generated SQL.

* Queries are **standard SELECT statements internally**, with AI handling interpretation.

### **Security and Data Governance**

* Data **never leaves the tenancy**.

* Integrates securely with **OCI Generative AI**, ensuring:

  * Confidentiality

  * Enterprise-grade security

* Compatible with fine-tuned models without exposing proprietary data.

### **Extensibility**

* **Pluggable AI Profiles**:

  * Choose provider and model per use case.

  * Assign specific schemas, tables, and views for processing.

* Supports **future-ready** enhancements with new or fine-tuned LLMs.

### **Key Advantages**

* No SQL expertise required to query enterprise data.

* Rapid app development using **AI-powered insights**.

* Maintains **full enterprise security**.

* Works natively within **Autonomous Database**, no external data transfer needed.

* Scalable and adaptable to multiple LLM providers.

### **Summary**

* **Select AI** simplifies database interaction using natural language.

* Converts user intent into SQL queries using LLMs.

* Integrates seamlessly with apps (like APEX) and visualization tools.

* Secure, future-proof, and pluggable for enterprise AI use cases.  
