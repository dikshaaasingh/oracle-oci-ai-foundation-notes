**OCI AI Foundations**

- **Introduction**


  ## **Why AI Skills Matter**

* **AI is reshaping work and careers.**  
   Companies use AI to drive growth, reduce costs, and deliver better customer value.

* **Managers prioritize AI aptitude.**  
   Hiring now focuses on candidates who understand or can use AI tools.

**AI at Work — Fast Adoption**

* 75% of knowledge workers use AI at work.

  * 46% started using AI in the last 6 months.

**AI Skills \> Experience**

* 66% of leaders won’t hire someone *without AI skills*.

  * 71% would prefer *a less experienced candidate with AI skills* over an experienced one without.

**AI is Mainstream**

* AI impacts *every industry and job role* — business, software, and creative fields.

  * Professionals are actively skilling up in AI.

  ## **Overview**

  ### **Two Core Domains:**

1. **AI Stack** (concepts & technologies)

2. **Oracle AI Stack** (Oracle’s AI services & tools)  
   

   ## **Domain 1: AI Stack**

| Level | Concept | Description |
| ----- | ----- | ----- |
| 1️⃣ | **Artificial Intelligence (AI)** | Machines mimicking human intelligence. |
| 2️⃣ | **Machine Learning (ML)** | Subset of AI — learns from data to predict outcomes or detect patterns. |
| 3️⃣ | **Deep Learning (DL)** | Subset of ML — uses *neural networks* to learn from complex data. |
| 4️⃣ | **Generative AI (GenAI)** | Creates new content — text, images, etc. |

*Modules 1–4* → Cover the entire AI Stack, tools, frameworks, and key terminologies.

## 

## 

## **Domain 2: Oracle AI Stack**

* Built on **high-performance Oracle AI infrastructure**.

* **Seamless integration** with Oracle ecosystem:

  * Databases: *Oracle Database*, *Autonomous Database*, *MySQL HeatWave*

  * SaaS Apps: *ERP*, *HCM*, *CX* with embedded AI/GenAI

* Not a “DIY” approach — Oracle provides a **complete AI environment**.

*Modules 5–7* →

* Oracle AI Stack Overview

* Generative AI Services

* Oracle AI Services (deep dive)  
    
- **AI Foundations**

	

- **INTRODUCTION TO AI**

**Definition:** AI is the ability of machines to imitate the cognitive and problem-solving abilities of human intelligence.

**Human Intelligence Capabilities**

* Learning new skills through observation and reasoning

* Understanding abstract concepts and applying logic

* Communication using language and interpreting nonverbal cues

* Real-time decision-making and handling complex objections

* Planning for short- and long-term goals

* Creativity in art, music, and innovation

**Artificial General Intelligence (AGI)**

* Machines replicating human sensory, motor, and intellectual abilities.

* Performs complex tasks independently.

* When AGI is applied to specific problems or tasks, it becomes **Artificial Intelligence (AI).**

**Examples of AI**

* Image recognition (identifying apple vs orange)

* Email spam detection

* Code generation

* Price prediction (e.g., used cars)

**Need for AI**

1. **Automation of Routine Tasks:**  
    Reduces repetitive business operations like credit approvals, insurance claims, and product recommendations.

2. **Enhanced Creativity and Assistance:**  
    AI acts as a creative partner—writing, designing, coding, or generating music and stories.

**Importance of AI**

* Handles massive data volumes beyond human capacity.

* Improves speed, accuracy, and decision-making efficiency.

**AI Domains and Examples**

* **Language:** Translation and natural language processing

* **Vision:** Image classification and recognition

* **Speech:** Text-to-speech and speech recognition

* **Recommendation Systems:** Product and content suggestions

* **Anomaly Detection:** Fraud detection and error spotting

* **Reinforcement Learning:** Self-driving cars, learning by reward

* **Forecasting:** Weather prediction and trend analysis

* **Generative AI:** Creating content such as images from text

- **AI \- Tasks and Data**

  ### **AI Tasks and Data Domains**

**Three Main Domains**

1. Language

2. Audio and Speech

3. Vision

   #### **1\. Language Domain**

**Types of Tasks**

* **Text-related:** Input is text; output varies by task.  
   Examples: language detection, entity extraction, key phrase extraction, translation.

* **Generative AI:** Creates new text content.  
   Examples: text generation (stories, poems), summarization, question answering, chatbots (e.g., ChatGPT).

**Text as Data**

* Text is sequential and made up of sentences and words.

* Words are converted to numbers using **tokenization**.

* Sentences are padded to equal length.

* **Embedding** represents similarity between words or sentences.

* Similarity can be measured using **dot similarity** or **cosine similarity**.

**Language AI Models**

* Models trained on large text datasets to perform NLP tasks.

* Common architectures:

  * **Recurrent Neural Networks (RNN):** Sequential processing with hidden states.

  * **Long Short-Term Memory (LSTM):** Retains context using gates.

  * **Transformers:** Parallel processing using self-attention for context understanding.

  #### **2\. Audio and Speech Domain**

**Types of Tasks**

* **Audio-related:** Input is speech or sound.  
   Examples: speech-to-text, speaker recognition, voice conversion.

* **Generative AI:** Generates new audio.  
   Examples: music composition, speech synthesis.

**Audio as Data**

* Audio is digitized as samples over time.

* **Sample rate:** Number of samples per second (e.g., 44.1 kHz \= 44,100 samples/sec).

* **Bit depth:** Bits per sample representing detail level.

* Single samples are not meaningful; multiple samples are correlated for analysis.

**Audio AI Models**

* Designed for sequential audio data.

* Common architectures:

  * RNN, LSTM, Transformers  
      
  * Variational Autoencoders (VAE)

  * Waveform Models

  * Siamese Networks

  #### **3\. Vision Domain**

**Types of Tasks**

* **Image-related:** Input is an image.  
   Examples: image classification, object detection, facial recognition.  
   Applications: security, biometrics, law enforcement, social media.

* **Generative AI:** Generates new images or visual content.  
   Examples: text-to-image generation, style-based images, 3D model creation.

**Images as Data**

* Images consist of **pixels** (grayscale or color).

* A single pixel has limited information; meaning comes from all pixels together.

* Task type determines required input and output.

**Vision AI Models**

* **Convolutional Neural Networks (CNN):** Detect visual patterns and hierarchical features.

* **YOLO (You Only Look Once):** Detects multiple objects in one pass.

* **Generative Adversarial Networks (GAN):** Generates realistic synthetic images.

  #### **4\. Other Common AI Tasks**

* **Anomaly Detection:**  
   Uses time-series data to identify unusual patterns (fraud detection, machine failure).

* **Recommendation Systems:**  
   Suggest products or content using data of similar users or items.

* **Forecasting:**  
   Uses time-series data to predict future outcomes (weather, stock prices).

- **Demo: AI**

**1\. Vision AI Service**

* Performs: **Image Classification, Object Detection, Text Detection, Document AI**.

**a. Image Classification**

* Detects objects and assigns **labels with confidence scores**.

* Example: detects zebra, vegetation, grassland, sky, etc.

**b. Object Detection**

* Draws **bounding boxes** around objects with labels and confidence scores.

* Detects multiple entities like **cars, people, fruits, bowls**, etc.

**c. Text Detection**

* Extracts **text from images**, including number plates, signs, and text in various fonts.

* Works line by line, detects text blocks and font variations.

**d. Document AI / Document Understanding**

* Extracts **text, key-value pairs, and tables** from documents (e.g., receipts).

* Identifies elements such as **transaction date/time, subtotal, tax, total, terminal ID**.

* Converts scanned data into structured format (tables and fields).

**2\. OCI AI Language Service**

* Supports **Text Analysis** and **Text Translation**.

**a. Text Analysis**

* **Language Detection:** identifies input language.

* **Text Classification:** classifies topic (e.g., Science & Technology).

* **Entity Extraction:** identifies entities (product, event, etc.) with confidence scores.

* **Key Phrase Extraction:** highlights important phrases.

* **Sentiment Analysis:** detects sentence-level and aspect-based sentiments (positive, neutral, negative).

* **PII Detection:** flags sensitive or identifiable information (e.g., names, dates).

**b. Text Translation**

* Translates text between multiple languages (e.g., English → French/Japanese).

* Option to **train custom translation models** using user-provided data.

- **AI vs ML vs DL**

**1\. Artificial Intelligence (AI)**

* Broad concept of machines performing human-like tasks.

* Example: Self-driving cars making decisions such as navigating traffic, detecting pedestrians.

**2\. Machine Learning (ML)**

* Subset of AI; algorithms learn from data to make predictions/decisions.

* Example: Spam email filter learning from user behavior and email content.

* **Algorithm:** set of rules or equations to learn from data.


**Types of ML:**

* **Supervised Learning:** Learns from labeled data.

  * Example: Credit card approval prediction from past data.

  * Process: Train → Predict new data outcomes.

* **Unsupervised Learning:** Finds patterns/clusters in unlabeled data.

  * Example: Retail customer segmentation, streaming service usage patterns, clustering fruits by nutrition.

* **Reinforcement Learning:** Learns by trial and error with feedback/rewards.

  * Example: Chess-playing AI, autonomous vehicles, robotics.

**3\. Deep Learning (DL)**

* Subset of ML; uses **deep neural networks** to learn complex patterns.

* Example: Image recognition software identifying cats or dogs.

* **Neural Networks:** Layers of interconnected nodes (“neurons”) that approximate complex functions from data.

**4\. Generative AI**

* Subset of ML; creates new content from learned patterns.

* Examples: Text (ChatGPT), images, audio, video.

* Applications: Content creation, innovation, problem-solving

  