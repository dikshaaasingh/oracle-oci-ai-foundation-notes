### **OCI AI Portfolio**

### **OCI AI Services Overview**

**1\. Oracle AI Ecosystem**

* Oracle integrates AI at all levels — infrastructure → data → AI services → business apps.

* OCI AI Services use enterprise data for specific business purposes.

* Focus: No infrastructure management; easy API-based consumption.

  ### **2\. Access Methods**

* **OCI Console** – Browser interface for AI & Data Science features.

* **REST API** – Programmatic access; requires coding knowledge.

* **Language SDKs** – Available for Java, Python, JS/TS, .NET, Go, Ruby.

* **Command Line Interface (CLI)** – Quick, script-free access to full features.

  ### **3\. OCI AI Service Categories**

Prebuilt ML models for easy app integration, customizable for better accuracy.

#### **a. Language Service**

* Processes unstructured text → extract insights.

* **Pretrained Models:**

  * Language detection

  * Sentiment analysis

  * Key phrase extraction

  * Text classification

  * Named Entity Recognition (NER)

  * PII detection

* **Custom Models:** Domain-specific NER & text classification.

* **Text Translation:** Neural Machine Translation across multiple languages.

  #### **b. Vision Service**

* Image upload → object detection/classification.

* **Pretrained Models:** Object detection, image classification, OCR.

* **Custom Models:**

  * Custom Object Detection (bounding boxes).

  * Custom Image Classification (identify domain-specific visuals).

  #### **c. Speech Service**

* Converts media files with speech → readable text.

* Outputs in **JSON** or **SRT** format.

* High accuracy for transcription tasks.

       **d. Document Understanding**

* Extracts and classifies data from documents.

* **Functions:**

  * OCR: Detects & recognizes text.

  * **Text Extraction:** Provides text with coordinates.

  * **Key-Value Extraction:** Extracts info (e.g., invoices, IDs).

  * **Table Extraction:** Maintains row-column structure.

  * **Document Classification:** Categorizes document types.

  #### **e. Digital Assistant**

* Platform for creating conversational AI interfaces.

* Handles:

  * User greetings & skill routing.

  * Listing and invoking skills.

  * Disambiguation, interruptions, and exit requests.

* Enables natural language-based task execution.

  ### **4\. Key Takeaways**

* OCI AI \= Ready-to-use AI for enterprises with minimal setup.

* Supports customization via APIs & SDKs.

* Covers all major AI domains: text, vision, speech, and conversation.

## **OCI Machine Learning Services Overview**

### **1\. Oracle AI & ML Layers**

* **Data** is the foundation for both AI and ML.

* **Top Layer:** Applications (AI consumption – apps, analytics, business processes).

* **Middle Layers:**

  * **AI Services:** Prebuilt models for specific use cases.

  * **ML Services:** Tools for custom model development.

* Focus: **OCI Data Science** — Oracle’s ML service for end-to-end model lifecycle.

### **2\. What is OCI Data Science?**

* Cloud service to **build, train, deploy, and manage ML models**.

* Supports **Python** and **open-source** tools.

* Serves **data scientists and teams** through the entire ML lifecycle.

* Provides **JupyterLab interface** with managed compute (CPU/GPU) and storage.

### **3\. Core Principles**

1. **Accelerated** –

   * Ready access to compute power.

   * Preinstalled open-source libraries and Oracle’s own ML library.

   * No infrastructure management.

2. **Collaborative** –

   * Shared assets, reproducibility, and auditability.

   * Team collaboration to reduce duplication and manage risk.

3. **Enterprise-Grade** –

   * Integrated with OCI’s security and IAM protocols.

   * Fully managed infrastructure (maintenance, patching, scaling).

### **4\. Key Features & Terminologies**

#### **a. Projects**

* Containers for organizing team work and assets (notebooks, models).

* Unlimited projects per tenancy.

* Collaborative workspaces for documentation and management.

#### **b. Notebook Sessions**

* Interactive **JupyterLab** environments.

* Preinstalled Python libraries; add others as needed.

* Run on managed computer (CPU/GPU).

* Used for **model building and training**.

#### **c. Conda Environments**

* Package & environment manager for Python.

* Quickly install/update dependencies and manage multiple environments.

#### **d. Accelerated Data Science (ADS) SDK**

* Oracle’s Python SDK for end-to-end ML workflow.

* Functions include:

  * Data access, exploration, visualization

  * AutoML training, evaluation, explainability

  * Integration with **Model Catalog** and **Object Storage**

#### **e. Models**

* Mathematical representations of business processes or data.

* Created within notebook sessions or projects.

#### **f. Model Catalog**

* Centralized repository for model artifacts.

* Stores metadata (versioning, Git info, training scripts).

* Enables sharing, tracking, and loading models across teams.

#### **g. Model Deployments**

* Deploy models from the catalog as **HTTP API endpoints**.

* Supports real-time predictions and web-based integration.

* Fully managed and scalable.

#### **h. Jobs**

* Define and run **repeatable ML tasks** on managed infrastructure.

* Used for automation and scheduled workflows.

### **5\. Summary**

* **OCI Data Science** simplifies the ML lifecycle — from **data to deployment**.

* Combines open-source flexibility with **Oracle-grade security and scalability**.

* Ideal for collaborative, enterprise-level ML development.

## **GPU and OCI AI Infrastructure**

### **1\. Importance of GPU in AI**

* **GPU (Graphics Processing Unit):** Specialized hardware designed for parallel computation.

* **Need in AI:**

  * AI and ML workloads require **massive repetitive calculations**.

  * Used heavily in **model training** and **inference**.

* **Parallel Computing:**

  * GPUs contain **thousands of lightweight cores** that process data simultaneously.

  * Enables **high-speed computation** and **large-scale data handling**.

  * Ideal for **batch inference** and serving **multiple simultaneous requests**.

### **2\. GPU vs CPU**

* **CPU:** Optimized for general-purpose sequential tasks.

* **GPU:** Optimized for **parallel tasks**; delivers **higher throughput** for ML and deep learning.

### **3\. Deep Learning Optimization**

* Modern GPUs are optimized for frameworks like:

  * **TensorFlow**

  * **PyTorch**

  * **ONNX Runtime**

* These use GPU-accelerated libraries (e.g., **CUDA**, **cuDNN**) for faster computation.

### **4\. NVIDIA GPU Architectures**

| GPU Model | Year | Architecture | Key Features |
| ----- | ----- | ----- | ----- |
| **A100** | 2020 | Ampere | Tensor Cores for fused multiply-accumulate ops; boosts deep learning. |
| **H100** | 2022 | Hopper | Transformer Engine for optimizing transformer model performance. |
| **H200** | 2024 | Hopper | Enhanced H100 with larger memory. |
| **B200 (Blackwell)** | 2025 | Blackwell | Designed for **large-scale AI and LLMs**. |
| **GB200 Superchip** | 2025 | Grace Blackwell | Combines Blackwell GPU with dual Grace CPUs; extreme performance for AI Cloud & HPC. |
| **NVIDIA Grace CPU** | — | — | Purpose-built for data centers and AI workloads. |

### **5\. OCI GPU Compute Offerings**

* **Expanded GPU lineup** for small to large AI workloads.

* Available/Upcoming options:

  * **OCI Compute with H100N (10,800 cores)**

  * **L40 GPU** – now generally available

  * **H200, B200 GPUs, and GB200 Superchips** – available for preorders (GA in 2025\)

  * **GPU Superclusters** – for large-scale AI and LLM training

**Performance Gains:**

* H200 superclusters → **4× performance** of H100.

* B200/GB200 → **8× performance** of H100 for AI workloads.

### **6\. Using GPUs in OCI Data Science**

* **OCI Data Science AI Quick Actions** enables:

  * Direct **deployment** of LLMs on GPU-powered VM or bare metal instances.

  * **Fine-tuning** of base models for custom inference.

* Supported models:

  * **Virtual LLM**

  * **Next-generation inference**

  * **Text-embedding inference containers**

### **7\. Summary**

* GPUs are essential for **scalable, high-speed AI computation**.

* **OCI AI Infrastructure** provides enterprise-grade GPU options for LLM training, fine-tuning, and inference.

* NVIDIA’s evolving architectures (Ampere → Hopper → Blackwell) drive next-gen AI performance on OCI.

### **OCI RDMA and Supercluster Architecture**

**1\. RDMA (Remote Direct Memory Access):**

* Enables **data transfer between machines without CPU involvement**.

* Provides **low latency, high bandwidth, and low CPU overhead**.

* Foundational for **OCI database, HPC, and GPU workloads**.

**2\. OCI Use of RDMA:**

* OCI uses **RoCE (RDMA over Converged Ethernet)** to enable RDMA on Ethernet fabric.

* Powers **ExaCS, Autonomous DB, HPC, and GPU clusters**.

**3\. RDMA Supercluster:**

* Built to support **massive GPU workloads (10,000–100,000 GPUs)**.

* Designed for **AI, ML, and large-scale parallel processing**.

**4\. GPU Node Architecture:**

* Each node: **8 NVIDIA A100 GPUs**, interconnected via **NVLink**.

* Each node connects to the network fabric at **1.6 Tbps (200 Gbps per GPU)**.

* **Non-blocking interconnect** ensures all GPUs can communicate simultaneously.

**5\. Network Fabric Design:**

* **Three-tier Clos network** for scalability.

* Scales to **100,000+ GPUs**.

* **Lossless RDMA network** — switches have high buffering and congestion control to prevent packet loss.

**6\. Latency and Performance:**

* Within a block: \~**6.5 microseconds** latency.

* Across blocks (worst-case): \~**20 microseconds** round trip.

* Still **10–20x faster** than typical cloud networks.

**7\. Placement Optimization:**

* **Control plane** automatically places workloads to balance **scale vs latency**.

* **Smaller workloads (DB, HPC)** → single block for minimal latency.

* **Large GPU workloads** → multiple blocks.

**8\. Network Locality & Placement Hints:**

* Provides **locality info** so workloads can optimize GPU communication paths.

* **85%+ of traffic remains local**, reducing latency and congestion.

* Leads to **higher throughput** and **fewer flow collisions**.

**9\. Key Optimizations Summary:**

* **Buffered switches** tuned for latency diameter → lossless network.

* **Placement mechanism** ensures low latency and high throughput.

* **Locality hints** help ML frameworks reduce cross-block communication.

**10\. Outcome:**  
 OCI’s RDMA Supercluster delivers:

* **Ultra-low latency** communication

* **Massive scalability** for AI/ML workloads

* **High throughput, lossless networking**, and **cost efficiency**

### **Responsible AI**

**1\. Definition & Need:**  
 AI is widely used today, but trust depends on ensuring it is **lawful, ethical, and robust** — minimizing harm and bias.

**2\. Core Principles of Trustworthy AI:**

* **Lawful:** Must comply with national and international laws.

* **Ethical:** Must align with human values and rights.

* **Robust:** Technically and socially sound to prevent unintended harm.

**3\. Legal Framework:**

* Existing laws govern AI use (e.g., data protection, safety).

* **Domain-specific rules** apply (e.g., medical device regulations in healthcare).

* Laws also **protect rights** and **enable fair innovation**.

**4\. Human Ethics & Fundamental Rights:**

* **Human dignity:** Respect mental and physical integrity.

* **Freedom:** Protect privacy, free expression, and choice.

* **Democracy:** AI must not undermine democratic systems.

* **Equality:** Avoid bias and ensure fairness.

* **Citizens’ rights:** Must remain protected during AI adoption.

**5\. Ethical Principles for AI:**

1. **Human oversight:** AI should assist humans, not replace them.

2. **No harm:** Must avoid physical or social harm.

3. **Transparency and fairness:** Decisions must be explainable and unbiased.

**6\. Responsible AI Requirements:**

* **Human-centric design:** Allow meaningful human control.

* **Safety and security:** Protect against malicious use.

* **Fairness:** Equal and just distribution of benefits and costs.

* **Explainability:** AI decisions should be interpretable to users.

**7\. Implementation Process:**

1. **Establish governance.**

2. **Define policies and procedures.**

3. **Monitor and evaluate compliance regularly.**

   * Roles involved: **Developers**, **Deployers**, **End users**.

**8\. Challenges in Healthcare AI:**

* **Bias:** Data imbalance (e.g., gender, race) affects fairness.

* **Transparency:** Complex algorithms reduce explainability.

* **Accountability:** Continuous evaluation needed to prevent harm.

## **OCI Data Science Demo – Key Steps**

### **1\. Overview**

* OCI Data Science is a **fully managed platform** for building, training, deploying, and managing ML models using Python and open-source libraries.

* Available under **Machine Learning stack → Analytics & AI → Data Science**.

* Supports team collaboration with projects, notebook sessions, jobs, pipelines, model catalog, and model deployments.

### **2\. Creating a Project**

1. Click **Create Project**.

2. Select **Compartment**, enter **Project Name** and **Description**.

3. Click **Create**.

4. Project will appear in **active state**.

### **3\. Notebook Sessions**

1. Click **Create Notebook Session**.

2. Select **Compartment** and provide a **Session Name**.

3. Choose **Compute Shape** (AMD, Intel, number of CPUs, RAM).

4. Configure **Block Storage, Network, Endpoint type**.

5. Click **Create**.

**Viewing existing sessions:**

* Shows **state, compute shape, network, endpoint**.

* Access **Notebook Session → JupyterLab environment**.

### **4\. Conda Environment**

1. Click **Environment Explorer** in Launcher.

2. Expand Conda environment to see **libraries and installation commands**.

3. Open **Terminal** in Launcher.

4. Run the Conda installation command.

5. Refresh to see new **kernel** in Jupyter.

6. Select kernel and create notebook.

Example: Python 3.8, general ML Conda environment.

### **5\. Data Science Workflow in Notebook**

1. **Import Libraries**

   * Include ADS library for Oracle Data Science automation.

2. **Load Dataset**

   * Example: Iris dataset.

3. **Prepare Data**

   * Split into **features** and **target variable**.

   * Split further into **train** and **test sets**.

4. **Train Model**

   * Example: `RandomForestClassifier`.

5. **Create Model Object**

   * Call `.prepare()` to generate artifacts for deployment automatically.

### **6\. Model Lifecycle Methods (ADS Library)**

| Method | Purpose |
| ----- | ----- |
| `.summary_status()` | Show workflow status, available methods, and required actions |
| `.verify()` | Simulate deployment; test predictions without actual deployment |
| `.save()` | Deploy model artifact to **Model Catalog** |
| `.deploy()` | Deploy model to **REST endpoint** |
| `.predict()` | Make predictions by invoking deployed model endpoint |

*   
  Steps **initiate → prepare → verify → save → deploy → predict** are followed sequentially.

* After `.save()`, model is visible in **Models tab**.

### **7\. Optional/Next Steps**

* Deploy model to REST endpoint using `.deploy()`.

* Generate predictions using `.predict()`.

* Explore **Jobs and Pipelines** for automated workflows.

### **Key Features Demonstrated**

* Fully managed JupyterLab environment.

* Choice of **CPU/GPU compute shapes**.

* Preconfigured Conda environments.

* ADS SDK automates ML workflow.

* Model catalog centralizes storage and versioning.

* REST endpoints for easy deployment.  
