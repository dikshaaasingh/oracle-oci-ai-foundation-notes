### **Deep Learning Foundations**

**Introduction**

**Definition**

* Subset of Machine Learning that uses Artificial Neural Networks (ANNs).

* Learn features automatically from raw data (e.g., image pixels).

**Key Idea**

* Unlike ML (where features are manually defined), DL extracts patterns/features automatically.

* Processes large data efficiently using parallel computation (batch processing).

**Applications**

* **Images:** classification, object detection, segmentation, facial recognition

* **Text:** translation, sentiment analysis, summarization, question answering

* **Audio:** music generation, speech-to-text

* **Generative tasks:** text-to-image, GANs, diffusion models

**Algorithms/Architectures**

* **CNN:** image tasks

* **RNN/LSTM/Transformers:** text tasks

* **Transformers, GANs, Diffusion Models:** generative tasks

**History**

* 1950s: Artificial neuron, perceptron, MLP

* 1980s: Backpropagation

* 1990s: CNN introduced

* 2000s: GPU introduction

* 2010s: Cheaper GPUs → deep learning boom (AlexNet, Deep-Q, generative models)

**Artificial Neural Network (ANN)**

* Inspired by human brain; composed of neurons (nodes).

* **Structure:**

  * Input layer (features)

  * Hidden layers (feature extraction)

  * Output layer (prediction)

* **Core components:**

  * **Neurons:** basic computational units

  * **Weights:** control connection strength

  * **Bias:** adds flexibility

  * **Activation function:** converts weighted sum into output

**Training Process**

* Inputs (e.g., 28×28 pixel images) → Input Layer

* Hidden layers learn patterns (edges, shapes)

* Output layer predicts digit (0–9)

* **Error correction:** Backpropagation adjusts weights to minimize error iteratively

**Outcome**

* Model learns to map inputs to correct outputs by optimizing weights through repeated training.

  ### **Deep Learning Models — Sequence Models**

**Purpose**

* Handle ordered data (sequences) and learn temporal dependencies to predict, classify, or generate new sequences.

* Common domains: NLP (translation, sentiment, generation), speech recognition, music generation, time-series forecasting, gesture recognition.

**Model Types for Sequences**

* **RNN (Recurrent Neural Network)**: maintains a hidden state that updates each time step, enabling the model to use prior context. Good for short-term dependencies.

* **LSTM (Long Short-Term Memory)**: an RNN variant with memory cells and gating to capture long-term dependencies and mitigate vanishing gradients.

* **Other choices**: GRU (Gated Recurrent Unit), Transformers (attention-based, excel at long-range dependencies and parallelism).

**RNN Architectures by Input–Output Pattern**

* **One-to-one**: standard feedforward (not sequence-specific).

* **One-to-many**: single input → sequence output (e.g., music generation).

* **Many-to-one**: sequence input → single output (e.g., sentiment classification).

* **Many-to-many**: sequence input → sequence output (e.g., machine translation, NER).

**Key Limitations**

* Vanilla RNNs struggle with long-term dependencies because of vanishing/exploding gradients during training.

**LSTM — Core Concepts**

* **Cell state**: internal memory that carries information across time steps.

* **Gates** (learned filters that control information flow):

  1. **Forget gate**: decides what information to drop from the cell state.

  2. **Input gate**: decides what new information to add to the cell state.

  3. **Output gate**: decides what part of the cell state to output as the hidden state.

* **Operation (per time step)**:

  1. Receive current input, previous hidden state, previous cell state.

  2. Compute gate values (forget, input, output).

  3. Update cell state using forget \+ input decisions.

  4. Produce new hidden state from updated cell state via output gate.

* LSTMs selectively remember/forget, enabling learning across long sequences.

**When to use which model**

* Short sequences or simple dependencies: RNN/GRU.

* Long sequences or when remembering distant context matters: LSTM or Transformer.

* For state-of-the-art NLP and large-scale sequence tasks: Transformer-family models (attention mechanism, parallelizable).

**Practical notes**

* Sequence inputs are typically tokenized/encoded (text → tokens; time series → windowed features).

* Choose loss and evaluation metrics per task (e.g., cross-entropy for classification, MSE for regression/forecasting).

* Use batching, sequence padding/truncation, and masking for variable-length sequences.

### **Deep Learning Models Convolutional — Neural Networks (CNN)**

**Overview of Deep Learning Architectures**

* **FNN / MLP**: Basic feedforward network for tabular data.

* **CNN**: Learns local spatial patterns in images/videos.

* **RNN**: Handles sequential/time-series data with feedback loops.

* **Autoencoders**: Unsupervised models for feature extraction & dimensionality reduction.

* **LSTM**: Specialized RNN for long-term dependencies.

* **GAN**: Generates synthetic data (images, audio, text).

* **Transformers**: Attention-based models for NLP and generative tasks.

**CNN Basics**

* Designed for grid-like data (images, videos).

* Preserves spatial structure of 2D data.

* Goal: extract hierarchical features while reducing data complexity without losing key information.

**CNN Architecture**

1. **Input Layer**: Accepts raw image data (2D array).

2. **Feature Extraction Layers** (repeated blocks):

   * **Convolutional Layer**: Applies filters (kernels) to detect features (edges, corners, textures).

   * **Activation Function (ReLU)**: Introduces non-linearity, enabling complex feature learning.

   * **Pooling Layer**: Reduces spatial size (dimensionality), highlights dominant features, prevents overfitting.

3. **Classification Layers**:

   * **Fully Connected (Dense) Layer**: Combines features for final decision.

   * **Softmax Layer**: Converts outputs to class probabilities.

   * **Dropout Layer**: Randomly drops neurons during training to reduce overfitting.

**Analogy (Robot Inspector)**

* Convolutional Layer → Blueprint detector (feature finder)

* Activation Function → Pattern highlighter

* Pooling Layer → Summarizer (reduces size)

* Fully Connected Layer → Expert (classifier)

* Softmax → Guess maker (predicts class)

* Dropout → Quality checker (prevents overreliance on features)

**Key Functions**

* **Convolution**: Local feature extraction via sliding filters.

* **Activation**: Enables non-linear mapping.

* **Pooling**: Spatial reduction for efficiency.

* **Dense & Softmax Layers**: Final classification.

* **Dropout**: Regularization to prevent overfitting.

**Limitations of CNN**

* Computationally expensive for large datasets.

* Prone to overfitting with limited data.

* Difficult to interpret (“black box”).

* Sensitive to small perturbations in input data.

**Applications of CNN**

* Image classification (cat vs dog).

* Object detection (bounding boxes).

* Image segmentation (pixel-level labeling).

* Face recognition and verification.

* Medical imaging (tumor detection, diagnosis).

* Autonomous driving (detecting signs, pedestrians, vehicles).

* Satellite and remote sensing (land cover, environment monitoring).

  ### **Demo Classification with Multilayer Perception**

**Objective**  
 Demonstrate how a deep learning model (MLP Classifier) can separate complex, non-linear data (two concentric circles).

**Dataset Creation (make\_circles \- sklearn)**

* **Function:** `make_circles(n_samples=300, noise=0.1, factor=0.5, random_state=seed)`

* **Description:** Generates 2D data points forming two concentric circles.

* **Parameters:**

  * `n_samples`: total number of points (e.g., 300 → 150 per class).

  * `noise`: adds randomness; higher noise → more scattered data.

  * `factor`: distance between inner and outer circles (0.5 → moderate gap).

  * `random_state`: ensures reproducibility.

* **Visualization:**

  * Inner circle → label **1** (green)

  * Outer circle → label **0** (red)

**MLP Classifier (Deep Learning Model)**

* Model: `MLPClassifier` (Multi-Layer Perceptron).

* Hidden layer: single layer with adjustable neurons.

* **Activation:** `relu` (introduces non-linearity).

* **Training:** continues until convergence (`max_iter` limit).

* **Effect of Hidden Neurons:**

  * **1 neuron:** Poor classification (almost all labeled 0).

  * **2 neurons:** Slight improvement, mixed misclassifications.

  * **3–4 neurons:** Complex decision boundaries, higher accuracy.

  * **5–6 neurons:** More refined and accurate boundaries.

**Interactive Visualization (update\_plot function)**

* Controlled via a **slider** to vary hidden layer size dynamically.

* Steps inside the function:

  1. **Create dataset:** `X` (coordinates), `y` (labels 0/1).

  2. **Initialize model:** `MLPClassifier(hidden_layer_sizes=(n,), activation='relu', random_state=...)`.

  3. **Train model:** `.fit(X, y)`.

  4. **Generate grid points:**

     * Create 100 × 100 mesh grid covering training data range.

     * Combine into feature pairs for prediction.

  5. **Predict labels:**

     * On grid points (for decision boundary).

     * On training points (for accuracy visualization).

  6. **Plot:**

     * `contourf()` → plots decision boundary.

     * Red \= class 0, Green \= class 1\.

     * Add labels, title, and show plot.

**Key Insights**

* Increasing **hidden neurons** allows the model to learn **non-linear decision boundaries**.

* Deep learning models like MLP can effectively separate **non-linearly separable data**.

* Visualization helps understand how complexity in architecture improves performance.

  