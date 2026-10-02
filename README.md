# Fashion MNIST CNN Classification

A Deep Learning project that uses a **Convolutional Neural Network (CNN)** built with **TensorFlow/Keras** to classify Fashion-MNIST images into 10 different clothing categories.

## 📌 Project Overview

The **Fashion-MNIST** dataset contains grayscale images of clothing and fashion items. The objective of this project is to build a CNN model that can automatically identify the category of a given image.

The model performs image preprocessing, CNN-based feature extraction, classification, and evaluation using accuracy, loss, classification report, confusion matrix, and incorrect prediction analysis.

## 🎯 Objective

* Load and explore the Fashion-MNIST dataset
* Visualize sample images and their labels
* Normalize image pixel values
* Prepare images for CNN input
* Build and train a CNN model
* Evaluate model performance on test data
* Analyze classification results using a confusion matrix
* Identify and visualize incorrect predictions

## 📊 Dataset

**Dataset:** Fashion-MNIST

The dataset contains 10 classes:

| Label | Class       |
| ----- | ----------- |
| 0     | T-shirt/top |
| 1     | Trouser     |
| 2     | Pullover    |
| 3     | Dress       |
| 4     | Coat        |
| 5     | Sandal      |
| 6     | Shirt       |
| 7     | Sneaker     |
| 8     | Bag         |
| 9     | Ankle boot  |

The images are grayscale and have a size of **28 × 28 pixels**.

The dataset is loaded directly using TensorFlow/Keras:

```python
datasets.fashion_mnist.load_data()
```

## 🛠️ Technologies Used

* Python
* TensorFlow
* Keras
* NumPy
* Matplotlib
* Scikit-learn
* Jupyter Notebook

## 🔄 Project Workflow

```text
Fashion-MNIST Dataset
        ↓
Data Exploration
        ↓
Image Visualization
        ↓
Pixel Normalization
        ↓
Reshape Images
        ↓
CNN Model
        ↓
Model Training
        ↓
Model Evaluation
        ↓
Classification Report
        ↓
Confusion Matrix
        ↓
Incorrect Prediction Analysis
```

## 🧹 Data Preprocessing

The pixel values originally range from **0 to 255**.

They are normalized to a range between **0 and 1**:

```python
X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0
```

The images are then expanded with a channel dimension so they can be used as CNN input:

```python
X_train = np.expand_dims(X_train, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)
```

The resulting image shape is:

```text
28 × 28 × 1
```

## 🧠 CNN Architecture

The CNN model consists of:

```text
Input Layer
     ↓
Conv2D (32 filters, 3×3)
     ↓
MaxPooling2D (2×2)
     ↓
Conv2D (64 filters, 3×3)
     ↓
MaxPooling2D (2×2)
     ↓
Flatten
     ↓
Dense (128 neurons)
     ↓
Dropout (0.5)
     ↓
Dense (10 neurons, Softmax)
```

### Model Configuration

**Optimizer:** Adam

**Loss Function:** Sparse Categorical Crossentropy

**Metric:** Accuracy

```python
cnn.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)
```

### Training Configuration

* Epochs: 15
* Batch size: 64
* Validation split: 10%

```python
history = cnn.fit(
    X_train,
    Y_train,
    epochs=15,
    validation_split=0.1,
    batch_size=64
)
```

## 📈 Model Evaluation

The model is evaluated using the test dataset:

```python
test_loss, test_accuracy = cnn.evaluate(X_test, Y_test)
```

The project also uses:

* Training vs. validation accuracy
* Training vs. validation loss
* Classification report
* Confusion matrix
* Incorrect prediction analysis

## 📋 Classification Report

The classification report provides:

* Precision
* Recall
* F1-score
* Support

```python
classification_report(
    Y_test,
    y_pred,
    target_names=class_names
)
```

## 🔲 Confusion Matrix

A confusion matrix is generated to understand how well the model performs for each of the 10 Fashion-MNIST classes.

```python
cm = confusion_matrix(Y_test, y_pred)
```

This helps identify which clothing categories are more frequently confused with one another.

## ❌ Incorrect Prediction Analysis

The project identifies images where the predicted class differs from the actual class:

```python
wrong_indices = np.where(Y_test != y_pred)[0]
```

The first 20 incorrect predictions are visualized along with their actual and predicted labels.

This helps analyze the types of images that are difficult for the CNN to classify.

## 🔍 Key Learning Outcomes

* Understanding CNN architecture for image classification
* Image normalization and reshaping
* Using convolution and pooling layers for feature extraction
* Using Dropout to reduce overfitting
* Training CNN models using TensorFlow/Keras
* Evaluating classification models using multiple metrics
* Interpreting confusion matrices
* Performing error analysis on incorrect predictions

## 📁 Repository Structure

```text
fashion-mnist-cnn-classification/
│
├── cnn_fashion_mnist.ipynb
├── README.md
└── requirements.txt
```

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/fashion-mnist-cnn-classification.git
```

Navigate to the project directory:

```bash
cd fashion-mnist-cnn-classification
```

Install the required libraries:

```bash
pip install -r requirements.txt
```

Run the notebook:

```bash
jupyter notebook cnn_fashion_mnist.ipynb
```

## 📦 Requirements

```text
tensorflow
numpy
matplotlib
scikit-learn
jupyter
```

## 🚀 Future Improvements

Possible improvements to the project include:

* Hyperparameter tuning
* Data augmentation
* Batch normalization
* Comparing different CNN architectures
* Saving and loading the trained model
* Deploying the model using Streamlit
* Creating an interface for uploading and classifying fashion images

## 👨‍💻 Author

**Ayush Khandare**


