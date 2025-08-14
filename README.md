Author

Tarun Kalisetti

# Spam Message Classification System

## Project Overview
This project is a machine learning pipeline to classify SMS messages as **HAM (not spam)** or **SPAM**. It demonstrates end-to-end workflow including text preprocessing, feature extraction, model training, evaluation, and prediction.

---

## Features
- Clean and preprocess text using **NLTK** (stopword removal, stemming, punctuation removal)
- Feature extraction using **TF-IDF vectorization**
- Classification using **Logistic Regression** and **Random Forest**
- Interactive prediction tool for new messages
- Saved models and vectorizer for future use

---

## Project Structure
spam-detection/
│
├── cleaned_data.csv # Preprocessed dataset
├── model.pkl # Trained ML model
├── vectorizer.pkl # TF-IDF vectorizer
├── train.py # Script to train ML models
├── predict.py # Script to predict new messages
├── textclean.py # Text cleaning utility
├── spam.csv # Original dataset
└── README.md # Project documentation



---

## How to Run

### 1. Install Dependencies
```bash
pip install pandas scikit-learn nltk joblib
2. Train Model
bash
Copy
Edit
python train.py
3. Predict Messages
bash
Copy
Edit
python predict.py
