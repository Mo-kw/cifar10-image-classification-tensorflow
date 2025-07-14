# CIFAR-10 Image Classification with TensorFlow 🚀

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Mo-kw/cifar10-image-classification-tensorflow/blob/main/Image_Classification_on_CIFAR_100_with_TensorFlow.ipynb)



This project demonstrates how to train and evaluate a Convolutional Neural Network (CNN) using TensorFlow to classify images from the CIFAR-10 dataset. It includes training, evaluation, visualization, and prediction on real images — and is fully runnable in **Google Colab**.

---

## 📂 Dataset

We use the **CIFAR-10** dataset, which contains 60,000 32x32 color images in 10 different classes:

- airplane ✈️  
- automobile 🚗  
- bird 🐦  
- cat 🐱  
- deer 🦌  
- dog 🐶  
- frog 🐸  
- horse 🐴  
- ship 🚢  
- truck 🚚  

---

## 💻 Run in Google Colab

> ✅ A ready-to-run Google Colab notebook is available!  
> It includes all cells to load the dataset, train the model, evaluate it, visualize results, and make predictions.

Click the **"Open in Colab"** badge at the top to get started instantly with GPU/TPU support.

---

## 🧠 Model Overview

- Uses **TensorFlow** and **Keras** with a pretrained CNN backbone (e.g., VGG-like).
- Resizes CIFAR-10 images from 32×32 to 224×224 to match input size of the pretrained network.
- Supports fine-tuning and saving/loading trained models.

---

## 🔧 Features

- ✅ Load and preprocess CIFAR-10 data  
- ✅ Resize images to 224×224  
- ✅ Train with GPU/TPU support  
- ✅ Evaluate using accuracy, precision, recall, F1-score  
- ✅ Plot confusion matrix  
- ✅ Visualize predictions on test samples  
- ✅ Load and test from saved model  
- ✅ Display model predictions on 10 test samples (one per class)

---

## 🧪 Evaluation Metrics

- **Accuracy**
- **Precision**
- **Recall**
- **F1-Score**
- **Confusion Matrix**
- Metrics are calculated on the test set using `scikit-learn`.

---

## 💾 Model Saving & Loading

Save:
```python
model.save("cifar10_model.h5")
