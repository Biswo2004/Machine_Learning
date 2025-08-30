# 📧 Spam Email/SMS Detector

A **Spam Detector** web application built with **Streamlit** that classifies messages as **Spam** or **Ham (Not Spam)** using machine learning (TF-IDF + Logistic Regression). This app allows users to check messages individually or in bulk and also provides performance evaluation of the trained model.

---

## 📝 Table of Contents

- [Features](#features)  
- [Dataset](#dataset)  
- [Technology Stack](#technology-stack)  
- [Installation](#installation)  
- [Usage](#usage)  
- [Project Structure](#project-structure)  
- [Model Details](#model-details)  
- [Screenshots](#screenshots)  
- [Contributing](#contributing)  
- [License](#license)  

---

## 🚀 Features

- **Single Message Detection:** Predict if a single email/SMS is spam or not.  
- **Bulk Detection:** Upload CSV files containing multiple messages for batch prediction.  
- **Model Evaluation:** View performance metrics including Accuracy, Precision, Recall, F1 Score, and Confusion Matrix.  
- **Interactive UI:** Simple, intuitive interface using Streamlit.  
- **Download Predictions:** Save bulk prediction results as a CSV file.  

---

## 📂 Dataset

This project uses the **SMS Spam Collection Dataset**, which is a collection of SMS labeled as `ham` (not spam) or `spam`.  

- Original dataset file: `SMSSpamCollection.csv`  
- Format: Tab-separated values (`\t`)  
- Columns:  
  - `label` → 'ham' or 'spam'  
  - `text` → SMS/email message  

The labels are converted to numerical values for the model: `ham = 0`, `spam = 1`.

---

## 🛠 Technology Stack

- **Python 3.x**  
- **Streamlit** – Web application interface  
- **Pandas** – Data processing  
- **Scikit-learn** – Machine learning model (TF-IDF + Logistic Regression)  
- **Seaborn & Matplotlib** – Visualizations  
- **CSV/Excel** – Input/Output for bulk predictions  

---

## ⚙️ Installation

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/spam-detector.git
cd spam-detector
