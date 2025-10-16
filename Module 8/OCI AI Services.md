### **OCI AI Services**

## **Demo: Language** 

OCI Language is part of **AI Services** under **Analytics & AI** in the Oracle Cloud Console. It provides a suite of **text analytics capabilities** that allow you to extract insights from text data easily.

### **Navigating the Console**

1. Go to **Analytics & AI → AI Services → Language**.

2. The Language page provides:

   * **Documentation links**

   * **API references**

   * **SDK guides**

   * **Blogs and tutorials**

### **Text Analytics Features**

Once you open **Text Analytics**:

1. **Text Input**

   * You can use default sample text or provide your own text for analysis.

2. **Language Detection**

   * Identifies the language of the text.

   * Provides **confidence score** for accuracy.

   * Example: “English” detected with high confidence.

3. **Text Classification**

   * Categorizes text into topics and subcategories.

   * Includes **probability/confidence score**.

   * Example: Science & Technology → Earth Sciences.

4. **Named Entity Recognition (NER)**

   * Highlights entities in the text:

     1. Locations

     2. Quantities

     3. Products

     4. Date/Time

   * Shows entity type and confidence.

   * Uses **color coding** to differentiate entity types.

5. **Key Phrase Extraction**

   * Lists important phrases from the text.

   * Useful for summarizing main ideas quickly.

6. **Sentiment Analysis**

   * Provides sentiment at **three levels**:

     1. **Document-level** – overall sentiment (positive, neutral, negative, mixed)

     2. **Aspect-based** – sentiment for specific aspects or entities in the text

     3. **Sentence-level** – sentiment for each individual sentence

   * Example: Mixed overall sentiment with negative aspect-level sentiments in some cases.

### **Summary**

* OCI Language Service allows you to quickly understand **text content**, **categorize it**, **extract entities and key phrases**, and **analyze sentiment** at multiple levels.

* The console provides an **interactive experience** to try these features without writing code.

* Insights can then be leveraged for applications such as:

  * Customer feedback analysis

  * Content categorization

  * Document summarization

  * Sentiment monitoring

## **Speech Intro**

OCI Speech converts **audio or video content into text** using advanced deep learning, providing accurate, timestamped, and readable transcriptions. No prior data science experience is needed.

### **Key Features**

1. **Automatic Transcription**

   * Converts speech in audio/video files directly into text.

   * Supports multiple languages:

     * English

     * Spanish

     * Portuguese

     * More languages coming soon.

   * Uses Oracle’s **acoustic and language models** for high accuracy.

2. **Processing**

   * Audio is **chunked** into smaller segments for fast processing.

   * Can transcribe hours of audio in **less than 10 minutes**.

   * Processes files directly from **Object Storage**.

3. **Batching Support**

   * Multiple audio/video files can be submitted in **a single API call**.

4. **Timestamped & Punctuated Text**

   * Each word and transcription comes with **confidence scores**.

   * Punctuation is added automatically for readability.

5. **SRT File Support**

   * Generates **SRT files** for closed captions.

   * Useful for videos and multimedia applications.

6. **Normalization**

   * Converts literal transcription into **human-readable text**.

   * Normalizes:

     * Numbers (words → numerals)

     * Dates and times

     * Addresses

     * URLs

   * Example: `"twenty-five"` → `"25"`.

7. **Profanity Filtering**

   * Options:

     * **Remove:** Replace profane words with asterisks.

     * **Mask:** Keep first letter, mask the rest.

     * **Tag:** Keep the word but tag it in the output.

### **Benefits**

* Quickly converts speech into actionable text.

* Easy integration into applications (transcription, captions, analytics).

* Enhances accessibility with readable text and SRT captions.

* High accuracy with minimal configuration.

## **Demo: Speech** 

### **1\. Navigate to Speech Service**

* Go to **Menu → Analytics & AI → AI Services → Speech**.

* Opens the **Speech service console**.

### **2\. Prerequisites**

* Ensure your **audio/video files are loaded into Object Storage**.

* You’ll need the **compartment** where your bucket resides.

### **3\. Create a Transcription Job**

1. Click **Create Transcription Job**.

2. Provide a **job name** (e.g., `Training`).

3. Select the **compartment** (preselected in your case).

4. Choose the **Object Storage bucket** containing your audio files.

5. Select the **file** to transcribe (e.g., `WAV` file).

6. Click **Run Job**.

### **4\. Monitor and Review Job**

* The job runs quickly (a few seconds for small files).

* Once complete, click the file to view **transcription results**.

### **5\. Transcription Output Highlights**

* **Punctuation added automatically** for readability.

* **Normalization applied**:

  * Numbers spelled out converted to numeric symbols (e.g., `"one hundred percent"` → `"100%"`).

* Multiple speakers handled correctly (e.g., support conversation).

* Generates **clean, human-readable text** ready for analysis or downstream applications.

## **Demo: Vision** 

OCI Vision is a **computer vision service** that works on images and provides two main capabilities:

### **1\. Image Analysis**

* **Object Detection**:

  * Detects objects in an image using **bounding boxes**.

  * Assigns **labels** to objects with **accuracy percentages**.

  * Can also **locate and extract text** visible in the image (e.g., signs, labels).

* **Image Classification**:

  * Assigns **classification labels** to the image by identifying its **major features**.

  * Useful for **categorizing images** based on content.

* **Custom Training**:

  * Beyond pretrained models, you can **retrain models** using your own datasets.

  * Enables **tailored models** for your specific use cases.

### **2\. Document AI**

* Processes documents to **extract structured information** from images of forms, receipts, or other documents.

## **OCI Vision Console Demo**

### **Accessing Vision**

1. Go to **OCI Console → Analytics & AI → Vision**.

2. Vision homepage provides:

   * Resources and documentation.

   * Quick access to features: **Image Analysis** (Image Classification & Object Detection) and **Document AI**.

   * Option to create **custom models** (not used in this demo).

### **1\. Image Classification**

* Default images available in the console.

* OCI Vision automatically assigns **tags** based on the image content.

* Examples:

  * Image of overhead power lines → Tags: `overhead power line, transmission tower, plant, sky, line`.

  * Image of London landmarks → Tags: `skyscraper, water, building, bridge, boat`.

**Insight:** Vision detects the **main features** of the scene accurately.

### **2\. Object Detection**

* Detects **specific objects** in an image and draws **bounding boxes** around them.

* Returns labels and confidence scores.

* Detects **text** within the scene (OCR capability):

  * License plates, logos, advertisements, bus routes, etc.

**Examples:**

* Street scene:

  * Front center: car

  * Back: bus

  * Sidewalk: people

  * Even partially visible cars are detected

* Low-resolution rooftop image:

  * Detects a person

  * Does **not** detect unusual objects like microwave antennas (model not trained for them)

* Cyclists on the road:

  * Detects persons and bicycles

  * Nested bounding boxes for wheels and riders

  * Highlights text features if visible (e.g., dashes on clothing)

### **Observations**

* Pretrained models are strong at detecting **common objects and text**.

* **Custom models** are recommended for specialized objects or rare items.

* Object detection can handle **complex scenes** with multiple people, vehicles, or overlapping objects.

## **Document Understanding**

### **Purpose**

* Designed for processing **document images** and PDFs.

* Can handle formats: **JPEG, PNG, TIFF**, and photographs containing text.

### **Key Features**

1. **Text Recognition (OCR)**

   * Extracts text from images.

   * Handles **complex scenarios**:

     * Handwritten text.

     * Tilted, shaded, or rotated documents.

   * Converts visual text into machine-readable form.

2. **Document Classification**

   * Classifies documents into **10 predefined types**.

   * Uses:

     * Visual appearance.

     * High-level features.

     * Extracted keywords.

   * Example use cases: invoice, receipt, resume.

3. **Language Detection**

   * Determines the **language of the text** based on **visual features**.

   * Works even when the text itself may be incomplete or noisy.

4. **Table Extraction**

   * Identifies **tables** in documents.

   * Extracts content in **structured tabular format**.

5. **Key-Value Extraction**

   * Identifies **specific fields** in documents.

   * Covers **13 common fields** and line items in receipts, e.g.:

     * Merchant name

     * Transaction date

   * Useful for automated processing and data entry.

**Summary:**  
 Document AI allows organizations to **automate document processing** by extracting text, classifying documents, detecting language, and identifying structured information like tables and key-value pairs, even in complex or handwritten documents.

Here’s a structured summary of your **OCI Vision – Document AI demo** based on what you just described:

## **Demo: Document Understanding**

### **1\. Receipt Example**

* **Language detection:** Correctly detected as **English**.

* **Text extraction:** All text on the receipt is highlighted and extracted.

* **Key-value extraction:** Automatically identifies specific fields for receipts:

  * Merchant name (e.g., Café)

  * Merchant address

  * Merchant phone number

  * Transaction date & time

* **Line item data:** Extracts items purchased in a **tabular format**, e.g., Americano, Water.

* **Use case:** Efficient expense processing by extracting all relevant information from a receipt automatically.

### **2\. Invoice Example**

* **Text extraction:** Highlights and extracts **all text**, including handwritten notes or stamped text.

* **Tabular extraction:**

  * Quantities, descriptions, unit prices, totals.

  * Extracted some **stamp information** into description columns without interfering with other data.

* **Key-value extraction:** Not applied for this invoice in this example.

* **Use case:** Automates accounts payable processing and extracts critical data even from imperfect scans.

**Key Takeaways from Demo:**

1. Document AI can handle **receipts, invoices, and complex document images**.

2. It extracts **raw text**, **key-value pairs**, and **tabular data** accurately.

3. Works with **handwritten, stamped, or imperfect scanned documents**.

4. Helps automate tasks like **expense processing** and **invoice management** efficiently.

