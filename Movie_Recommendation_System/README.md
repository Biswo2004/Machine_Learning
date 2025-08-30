# 🎬 Movie Recommendation System

A **Movie Recommender** web application built with **Streamlit** that suggests similar movies based on **genres, release year, and rating** using **TF-IDF and cosine similarity**. Users can enter a movie name and get a list of recommendations displayed in an interactive interface.

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

- **Search by Movie Name:** Type a movie name and get relevant recommendations.  
- **Top-N Recommendations:** Default top 5 similar movies based on TF-IDF similarity.  
- **Interactive UI:** Modern, clean, and responsive interface using Streamlit.  
- **Handles Missing Data:** Missing fields in the dataset are handled gracefully.  

---

## 📂 Dataset

This app uses a **CSV dataset** named `merged_dataset.csv` with the following columns:

- `name` → Movie title  
- `year` → Release year  
- `genres` → Movie genres (comma-separated)  
- `rating` → IMDb or user rating  

**Note:** You can replace `merged_dataset.csv` with your own dataset, but it must have the same column structure.

---

## 🛠 Technology Stack

- **Python 3.x**  
- **Streamlit** – Web interface  
- **Pandas** – Data processing and manipulation  
- **Scikit-learn** – TF-IDF vectorization and cosine similarity computation  

---

## ⚙️ Installation

1. **Clone the Repository**  
   ```bash
   git clone <your-repo-url>
   cd <repo-folder>

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
