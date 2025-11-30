# CP33_TEAM_B_Evoastra_Major_Project
Image Captioning Using CNN–LSTM with Attention Major Project by Team B (CP33) under Evoastra Data Science Internship — Image Caption Generator Trained on COCO train 2017 118k/18 GB and annotation_trainval2017 [ 241 Mb ] Dataset

A Deep Learning Project Combining Computer Vision and Natural Language Processing

COCOC DATASET OFFICIAL WEEBSITE - https://cocodataset.org/#download

COCO train 2017 118k/18 GB - http://images.cocodataset.org/zips/train2017.zip

annotation_trainval2017 [ 241 MB ] Dataset - http://images.cocodataset.org/annotations/annotations_trainval2017.zip

# Presentation
👉 View the presentation on Canva:
[Open Canva Presentation](https://www.canva.com/design/DAG4xS4y30k/v5eE32S8iKl_5hmz4sjZlw/edit)

## project Title 
**Image Captioning Using CNN–LSTM with Attention**

## 🎯 Project Objectives 
- To develop an end-to-end Image Captioning System that generates human-like captions for images using deep learning.
- To integrate Computer Vision (CNN) and Natural Language Processing (LSTM) into a unified model architecture.
- To extract meaningful visual features using pre-trained InceptionV3 for efficient and accurate representation of images.
- To build an Encoder–Decoder model with Attention that improves contextual understanding and caption quality.
- To preprocess and structure image–caption data for effective model training and generalization.
- To evaluate caption quality using BLEU Score and compare model-generated captions with ground-truth captions.
- To provide a foundational implementation for advanced Vision + NLP tasks such as image retrieval, and multimodal AI applications.


 ## 📌 Project Overview
- This project builds an end-to-end Image Captioning System that automatically generates descriptive captions for images.
- The system integrates:
- InceptionV3 CNN Encoder for image feature extraction
- LSTM Decoder with Attention for sequence generation
- Tokenizer-based caption preprocessing
- BLEU Score evaluation for caption quality
- This project demonstrates the power of combining Computer Vision + NLP to create intelligent multimodal AI applications.

## Project Requirements
 ** 1. Software Requirements
- Python 3.8+
- TensorFlow / Keras – Model building, training, and attention mechanism
- NumPy – Numerical operations
- Pandas – Caption and dataset handling
- Matplotlib / Seaborn – Visualizing results
- NLTK / Keras Tokenizer – Text preprocessing, tokenization
- VS Code – Development environment

**2. Hardware Requirements
- GPU Recommended (NVIDIA CUDA-supported)
- Training image captioning models is compute-heavy
- Minimum 8 GB RAM (16 GB recommended)
- Minimum 10 GB free storage for
- MS COCO dataset
- Extracted image features
- Trained models

**3. Dataset Requirements
- MS COCO Dataset (2014/2017)
- Images (~80k training, ~40k validation)
- Caption annotations (5 captions per image)

## 🧠 Steps Followed
1. Data  Collection Preparation
2.Feature Extraction
3. Model Training
4. Model Evaluation


## 📁 Data Preparation

** Data Source**
Dataset Link :
Training : http://images.cocodataset.org/zips/train2017.zip
Validation : http://images.cocodataset.org/annotations/annotations_trainval2017.zip

**1. Image Preprocessing**
- Resize images to 299 × 299 pixels
- Normalize pixel values to the range [0–1]
- Convert images to arrays (NumPy format)

**2. Caption Preprocessing**
Steps include:
- Lowercasing and removing special characters
- Tokenization using Keras Tokenizer
- Converts words into unique integer indices.
- Padding sequences to a fixed length
- Ensures all captions in a batch have uniform size.
- Adding special tokens
- <start> → Marks beginning of the caption
- <end> → Marks the end of the caption

** 3. Output of Data Preparation**
- Normalized images ready for CNN feature extraction
- Tokenized and padded caption sequences
- Vocabulary dictionary (word → integer mapping)
- Training-ready input/output pairs for the decoder

## Model Training
** Training Overview
- The decoder (LSTM) and attention layers are trained to generate captions conditioned on image features extracted by the encoder. Training follows a supervised sequence-to-sequence setup where the model learns   to predict the next word in a caption given previous words and the visual context.

** Training Workflow
- Load precomputed image features (extracted from the encoder) and corresponding tokenized caption sequences.
- Create input–output pairs for the decoder: each partial caption sequence serves as input and the following word is the target.
- Batch the data and feed (image_features, input_sequence) to the model to predict the target token.
- Compute loss and backpropagate gradients to update trainable decoder and attention parameters.
- Validate on a held-out set each epoch and save checkpoints for the best validation performance.

** Pre-trained Model: InceptionV3 (Encoder)
- Using a pre-trained CNN (trained on ImageNet) provides rich visual feature representations learned from millions of images. This transfer learning approach:
- Reduces training time (no need to learn low-level filters from scratch).
- Improves generalization by leveraging robust, high-level visual features.
- Stabilizes learning of the decoder since visual embeddings are informative from the start.

** Key Hyperparameters
- Batch size: e.g., 64 (tune according to GPU memory)
- Learning rate: e.g., 1e-3 initial; use schedulers (ReduceLROnPlateau) or manual decay
- LSTM units: e.g., 256–512 (depends on vocabulary size and dataset scale)
- Embedding dimension: e.g., 256–512
- Maximum caption length: set according to tokenized data (truncate longer captions)
- Epochs: typically 10–30, monitor validation loss / BLEU to decide

** Monitoring & Checkpointing
- Track training and validation loss, token accuracy, and BLEU scores periodically.
- Use ModelCheckpoint to save the best model weights (by validation loss or BLEU).
- Implement EarlyStopping to halt training when validation performance stops improving.
- Log metrics with TensorBoard or a simple CSV logger for reproducibility.

## 📊 Model Evaluation
Model evaluation aims combination of quantitative metrics and qualitative analysis is used to ensure the model not only produces grammatically correct captions but also identifies objects and context effectively.

**1. Quantitative Evaluation — BLEU Score
- BLEU Measures:
- BLEU-1: Unigram (word-level) accuracy
- BLEU-2: Bigram consistency
- BLEU-3: Trigram fluency
- BLEU-4: Overall sentence-level similarity
- Higher BLEU scores indicate better alignment with human-like descriptions.

** 2. Validation During Training
- During training, evaluation is performed periodically to monitor performance:
- Compute BLEU scores on the validation subset
- Track loss, token accuracy, and caption coherence
- Save the best-performing model weights using ModelCheckpoint
- Use EarlyStopping to prevent overfitting
- This helps ensure the model generalizes well beyond the training images.

## ✔️ Results
- The Image Captioning system successfully demonstrates the ability to generate meaningful and context-aware captions for diverse images. Key outcomes include:
- Accurate object detection in most common scenes (people, animals, vehicles, food items, indoor/outdoor environments).
- Coherent and grammatically correct captions generated by the LSTM decoder.
- Attention mechanism significantly improves performance, enabling the model to focus on relevant image regions.
- Consistent BLEU scores across validation sets, confirming stable caption quality.
- Strong generalization on unseen images, particularly when image structure is simple and objects are clearly identifiable.

## 📚 Learnings
1. Vision and Language Integration
Understanding how image features (CNN) and textual features (LSTM embeddings) work together in a unified architecture.
2. Transfer Learning
Using a pre-trained CNN (InceptionV3) to extract deep visual features, reducing training time and improving accuracy.
3. Sequence Modeling
Building and training an LSTM-based decoder that predicts captions word-by-word.
4. Attention Mechanism
Learning how attention helps the model focus on relevant regions in an image during caption generation.
5. Evaluation Techniques
Using BLEU scores and qualitative inspection to measure caption quality effectively.

## ⚠️ Challenges & 💡 Solutions
1. Complex Scene Understanding
** Challenge:
- Images with multiple overlapping or small objects often resulted in incomplete or overly short captions.
- The decoder struggled to identify detailed relationships between objects.
** Solution:
- Incorporated an Attention Mechanism to allow the model to focus on specific regions of the image.
- This significantly improved caption clarity, detail, and context understanding.

2. Limited Vocabulary & Rare Words
** Challenge:
- The model performed poorly on:
- Uncommon object names
- Domain-specific terminology
- Rare categories in the MS COCO dataset
** Solution:
Applied vocabulary optimization techniques:
- Removed extremely rare words to reduce noise
- Normalized and standardized caption length
- Improved tokenization and padding strategies
- This enhanced the model’s ability to generalize while keeping the vocabulary manageable.

3. Overfitting During Training
** Challenge:
- The model showed:
- High training accuracy
- Lower validation accuracy
- indicating overfitting, especially to frequent caption structures.
** Solution:
- Implemented regularization techniques including:
- Dropout layers in the decoder
- EarlyStopping to halt training when validation loss stagnated
- Learning rate schedulers (LR decay) to stabilize convergence
- These measures improved generalization and reduced overfitting.

4. Computational Constraints
** Challenge:
- Training on large datasets like MS COCO required:
- Significant GPU processing power
- Long training time
- Efficient memory utilization
** Solution:
- Enabled precomputation of CNN features using InceptionV3 and cached them on disk.
- This approach:
- Reduced GPU load
- Shortened training cycles
- Increased overall workflow efficiency


---

## 👥 Team Members
**Team B – CP33**  
- **MOHAMMAD SHARJEEL YAZDANI** – Team Lead
- **Rishabh Tanpure** – Co-Lead 1  
- **Samradnyi Kale** – Co-Lead 2
- **Sakshi Giglani** – Member
- **Monu Kumar Jha** – Member
- **Sriya Sahu** – Member
---

## 🏢 Organization
**Evoastra Data Science Internship Program – 2025**  
🔗 [https://www.evoastra.in](https://www.evoastra.in)

---



            


