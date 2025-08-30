# 🖼️ MNIST Digit Classifier with CNN

A **MNIST Digit Classifier** web application built with **Streamlit** and **TensorFlow/Keras**. Users can draw digits (or sequences of digits) on a canvas, and the app predicts them using a Convolutional Neural Network (CNN). The app also allows training/retraining the model interactively and visualizes training progress.

---

## 📝 Table of Contents

- [Features](#features)  
- [Dataset](#dataset)  
- [Technology Stack](#technology-stack)  
- [Installation](#installation)  
- [Usage](#usage)  
- [Project Structure](#project-structure)  
- [How It Works](#how-it-works)  
- [Screenshots](#screenshots)  
- [Contributing](#contributing)  
- [License](#license)  

---

## 🚀 Features

- **Draw Digits:** Draw single or multiple digits on an interactive canvas.  
- **CNN Predictions:** Predict digits in real-time using a trained CNN model.  
- **Train/Retrain:** Train the CNN model interactively from the sidebar.  
- **Segment Multiple Digits:** Handles sequences of digits drawn side-by-side.  
- **Probability Visualization:** Displays prediction probabilities for each digit.  
- **Compact Training Graph:** Visualizes training and validation accuracy.  

---

## 📂 Dataset

This app uses the **MNIST dataset** of handwritten digits:

- **Training images:** 60,000  
- **Test images:** 10,000  
- **Image size:** 28x28 pixels, grayscale  
- **Labels:** 0–9  

The dataset is automatically loaded using Keras (`tensorflow.keras.datasets.mnist`).

---

## 🛠 Technology Stack

- **Python 3.x**  
- **Streamlit** – Web interface  
- **TensorFlow / Keras** – CNN model  
- **NumPy** – Numerical computations  
- **Matplotlib** – Probability and training visualizations  
- **OpenCV** – Image preprocessing and segmentation  
- **streamlit_drawable_canvas** – Drawing canvas for digits  

---

## ⚙️ Installation

1. **Clone the Repository**  
   ```bash
   git clone https://github.com/yourusername/mnist-cnn-streamlit.git
   cd mnist-cnn-streamlit
---

**Create & Activate a Virtual Environment:** 

`python -m venv venv && source venv/bin/activate` 
- *(Windows: `venv\Scripts\activate`)*
- *(linux/macos: `source venv/bin/activate`)*

---

**Install Dependencies:**
- `pip install -r requirements.txt`

---
## ▶️ Usage

1. **Run the Streamlit app:**
  - `streamlit run app.py`
