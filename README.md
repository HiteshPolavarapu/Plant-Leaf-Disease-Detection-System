# 🌿 Plant Leaf Disease Detection System

An AI-powered system that uses a **Convolutional Neural Network (CNN)** in TensorFlow to analyze leaf images for early disease diagnosis. This tool empowers farmers to improve crop yield and quality, reduce losses, and contribute to agricultural sustainability.



---

## 📋 Table of Contents

- [Project Workflow](#-project-workflow)
- [Model Performance and Evaluation](#-model-performance-and-evaluation)
- [Getting Started](#-getting-started)
- [Technology Stack](#️-technology-stack)

---

## 🎯 Project Workflow

This project implements an end-to-end system for classifying plant leaf diseases using a CNN. The process is broken down into several key stages.

### 1. Setup and Environment
The initial phase involves setting up the environment and preparing the dataset.
* **Import Libraries:** Imports essential Python libraries, including **TensorFlow**, **Keras**, **Matplotlib**, **Seaborn**, and **NumPy**.
* **Mount Google Drive:** Connects to Google Drive to access the dataset stored in the cloud.
* **Unzip Dataset:** Extracts the compressed dataset into the local environment, making image files accessible for training.

### 2. Data Preprocessing and Augmentation
This stage prepares the image data for the model.
* **ImageDataGenerator:** Utilizes Keras's `ImageDataGenerator` for two critical tasks:
    * **Rescaling:** Normalizes pixel values from the `[0, 255]` range to `[0, 1]` to improve training stability and performance.
    * **Data Augmentation:** Applies random transformations (rotation, zoom, shear, flips) to the training images. This artificially expands the dataset, which is a crucial technique for preventing **overfitting** and improving the model's ability to generalize to new, unseen images.
* **Data Loading:** The `flow_from_directory` method loads images, resizes them to a uniform dimension (e.g., `224x224`), and organizes them into batches.

### 3. CNN Model Architecture
The core of the project is a **Convolutional Neural Network (CNN)**, built as a `Sequential` model.
* **Convolutional & Pooling Layers (`Conv2D`, `MaxPooling2D`):** These layers work together to automatically extract features from the images, such as textures, patterns, and shapes relevant to diseases.
* **Flatten Layer:** Converts the 2D feature maps into a 1D vector.
* **Dense Layers:** Fully-connected layers that perform the classification based on the extracted features. The final `Dense` layer uses a **softmax** activation function to output a probability score for each disease class.
* **Dropout Layer:** A regularization technique that randomly deactivates a fraction of neurons during training to prevent overfitting.

### 4. Model Training
The model is configured and trained in this section.
* **Compilation:** The `model.compile()` step configures the training process by specifying the **Adam optimizer** and the **categorical crossentropy** loss function, which are standard for multi-class image classification.
* **Training:** The `model.fit()` function trains the model by iterating over the dataset for a specified number of **epochs**. During this process, the model learns to map the input images to their correct labels.

### 5. Evaluation and Prediction
After training, the model's performance is evaluated.
* **Performance Visualization:** Generates plots of accuracy and loss over epochs to diagnose the training process.
* **Prediction:** A function is defined to preprocess a new, unseen image and use the trained model to predict the disease class.
* **Model Saving:** The final trained weights are saved to a file (e.g., `model.h5`), allowing the model to be deployed without retraining.

---

## 📊 Model Performance and Evaluation

### Key Performance Metrics
* **Accuracy:** The overall percentage of correct predictions.
* **Precision:** Measures the accuracy of positive predictions. High precision indicates a low false-positive rate.
* **Recall (Sensitivity):** Measures the model's ability to identify all relevant cases. High recall indicates a low false-negative rate.
* **F1-Score:** The harmonic mean of precision and recall, providing a single metric that balances both.

### Visualizing the Results
* **Accuracy and Loss Plots:** These graphs are essential for visualizing the model's learning progress and identifying issues like overfitting.
* **Confusion Matrix:** A grid that provides a detailed breakdown of classification results, showing exactly which classes the model is confusing.

### Practical Demonstration
The ultimate test involves feeding new, unseen images to the model and evaluating its predictions and confidence scores, proving its value as a real-world diagnostic tool.

---

## 🚀 Getting Started

To get a local copy up and running, follow these steps.

1.  **Clone the repository:**
    ```sh
    git clone [https://github.com/HiteshPolavarapu/Plant-Leaf-Disease-Detection-System.git]
    ```
2.  **Install dependencies:**
    ```sh
    pip install -r requirements.txt
    ```
3.  **Run the notebook:**
    Open the `.ipynb` file in Jupyter Notebook or Google Colab and execute the cells.

---

## 🛠️ Technology Stack

* **TensorFlow & Keras**
* **Python**
* **Pandas**
* **NumPy**
* **Matplotlib & Seaborn**
* **Scikit-learn**
* **Google Colab**
