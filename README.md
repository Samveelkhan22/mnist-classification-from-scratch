# MNIST Classification from Scratch

## 📌 Project Overview

This project implements core Machine Learning classification algorithms **from scratch** and evaluates them on the MNIST handwritten digits dataset.

The goal of this project is to understand how classification models work internally without relying on high-level ML libraries for training.

Implemented Models:
- Perceptron  
- Gaussian Naive Bayes  
- Logistic Regression (Softmax-based)

The project follows a modular, object-oriented ML framework with custom training loops, evaluation metrics, and hyperparameter management.

---

## 🧠 Dataset

The project uses the MNIST dataset:

- 60,000 training images  
- 10,000 testing images  
- 10 classes (digits 0–9)  
- 28×28 grayscale images  

Each image is flattened into a feature vector before training.

---

## 🏗 Project Architecture

### 1️⃣ Base Model Framework
- `BaseModel` (Abstract class)
- `ClassificationModel` (Abstract subclass)

These enforce:
- `fit(X, y)`
- `predict(X)`

---

### 2️⃣ Implemented Models

#### 🔹 Perceptron
- Gradient-based weight updates
- Configurable learning rate
- Multiple training epochs
- Works well for linearly separable data

#### 🔹 Gaussian Naive Bayes
- Class-wise mean and variance estimation
- Gaussian likelihood assumption
- Variance smoothing for numerical stability

#### 🔹 Logistic Regression
- Multiclass softmax implementation
- Gradient descent optimization
- Cross-entropy-based training
- Configurable learning rate and epochs

---

### 3️⃣ Data Preparation Pipeline

Custom data processing classes:
- `DataPreparation`
- `MNISTDataPreparation`

Features:
- Train/Validation split
- Optional binarization
- Data transformation
- Conversion to NumPy arrays

---

### 4️⃣ Hyperparameter Management

Centralized configuration via:
`HyperParametersAndTransforms`

Used for:
- Model hyperparameters
- Preprocessing configuration

---

### 5️⃣ Model Runner

The `RunModel` class handles:
- Model building
- Training
- Validation
- Accuracy computation
- Confusion matrix generation

---

## 📊 Training & Evaluation

To run evaluation:

```python
if __name__ == "__main__":
    run_eval()
```

The pipeline:
1. Loads MNIST dataset
2. Applies preprocessing
3. Trains each model
4. Computes validation accuracy
5. Evaluates performance

---

## 🚀 Key Learning Outcomes

- Implementing ML algorithms from scratch
- Understanding gradient descent mechanics
- Implementing softmax manually
- Computing Gaussian likelihoods
- Designing modular ML systems
- Building structured evaluation pipelines

---

## 🛠 Technologies Used

- Python
- NumPy
- Scikit-learn (metrics only)
- Object-Oriented Programming

---

## 🎯 Skills Demonstrated

- Machine Learning fundamentals
- Mathematical model implementation
- Clean code architecture
- Data preprocessing
- Model validation & evaluation

---

## 📌 Project Philosophy

This project focuses on understanding machine learning models beyond black-box usage by manually implementing optimization, probability estimation, and prediction logic.
