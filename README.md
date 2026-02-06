# Spam Email Classifier

This project is a Machine Learning based spam email classifier built using Python and Scikit-learn.

## Features
- Classifies emails as Spam or Not Spam
- Uses TF-IDF for text vectorization
- Uses Naive Bayes algorithm
- Command-line based prediction

## Tech Stack
- Python
- Pandas
- Scikit-learn

## How It Works
1. Email text is converted into numerical features using TF-IDF
2. Naive Bayes model is trained on labeled data
3. Model predicts whether a new email is spam or not

## How to Run
```bash
pip install -r requirements.txt
python train.py
python predict.py
