# Plant-Leaf-Disease-Detection-System
A Plant Leaf Disease Detection System using a Convolutional Neural Network (CNN) in TensorFlow. This AI-powered tool analyzes leaf images for early disease diagnosis, empowering farmers to improve crop yield and quality, reduce losses, and contribute to agricultural sustainability.

Project Workflow Explained 📝
This notebook implements an end-to-end system for classifying plant leaf diseases using a Convolutional Neural Network (CNN). The process is broken down into several key stages.

1. Setup and Environment
The first part of the notebook sets up the necessary environment and libraries.

Import Libraries: It imports essential Python libraries for deep learning and data handling, including TensorFlow and Keras for building the model, Matplotlib and Seaborn for plotting results, and NumPy for numerical operations.

Mount Google Drive: The code connects to Google Drive to access the dataset, which is assumed to be stored there.

Unzip Dataset: It unzips the compressed dataset into the Colab environment's temporary storage, making the image files accessible for training.

2. Data Preprocessing and Augmentation
This stage prepares the image data for the model. Raw images cannot be fed directly into a neural network; they must be formatted correctly.

ImageDataGenerator: The notebook uses Keras's ImageDataGenerator to process the images. This powerful tool does two critical things:

Rescaling: It normalizes the pixel values of the images from the range [0, 255] to [0, 1], which helps the model train more efficiently.

Data Augmentation: For the training set, it applies random transformations to the images (like rotation, zooming, shearing, and horizontal flipping). This creates slightly modified versions of existing images, artificially expanding the dataset. This technique is crucial for preventing overfitting and helping the model generalize better to new, unseen images.

Data Loading: The flow_from_directory method is used to load images from their respective folders. It automatically resizes all images to a uniform size (e.g., 224x224 pixels) and organizes them into batches.

3. CNN Model Architecture
The core of the project is the Convolutional Neural Network (CNN). The notebook defines a Sequential model, which is a linear stack of layers.

Convolutional Layers (Conv2D): These are the primary building blocks. They act as feature detectors, scanning the images to identify patterns like edges, textures, colors, and shapes that are characteristic of certain diseases. The model uses multiple convolutional layers to learn increasingly complex features.

Pooling Layers (MaxPooling2D): After each convolutional layer, a pooling layer is used to downsample the image. This reduces the computational complexity and makes the detected features more robust to variations in position.

Flatten Layer: This layer converts the 2D feature maps from the convolutional/pooling layers into a single 1D vector, preparing the data for the final classification stage.

Dense Layers: These are fully-connected neural network layers.

The first Dense layer acts as a classifier, processing the features from the flatten layer.

The final Dense layer has a softmax activation function. It outputs a probability score for each of the 38 possible classes (diseases), indicating the model's confidence in its prediction.

Dropout Layer: This layer is a regularization technique used to prevent overfitting by randomly "dropping out" (ignoring) a fraction of neurons during training.

4. Model Training
This section covers the process of training the model.

Compilation: The model.compile() step configures the model for training. It specifies the:

Optimizer (adam): An efficient algorithm that adjusts the model's internal weights to minimize error.

Loss Function (categorical_crossentropy): A function that measures how different the model's predictions are from the actual labels. The model's goal is to minimize this loss.

Training (model.fit): This is the main training loop. The model iterates over the training dataset for a specified number of epochs. In each epoch, it learns the relationships between the images and their labels, adjusting its weights to improve its accuracy. The performance is monitored on the validation set at the end of each epoch.

5. Evaluation and Prediction
After training, the notebook evaluates the model's performance and uses it to make predictions.

Performance Visualization: It plots graphs of the training and validation accuracy and loss over epochs. These plots are essential for diagnosing the training process and checking for issues like overfitting.

Prediction on New Images: A function is defined to take a new, unseen image, preprocess it in the same way as the training data, and feed it to the trained model. The model then returns the predicted disease class, demonstrating its practical application.

Model Saving: Finally, the trained model's weights are saved to a file (.h5 or .keras). This allows the model to be reused later for inference in a web or mobile application without needing to be retrained.
