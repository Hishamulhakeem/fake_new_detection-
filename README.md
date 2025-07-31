# Fake News Detection Project

This project aims to classify news articles as fake or real using machine learning models, specifically Logistic Regression and Decision Tree Classifier. The dataset consists of news articles labeled as fake or real, and the goal is to predict the veracity of a given news article.

## Files

### 1. `faknews.py`
This script preprocesses the dataset and performs the following tasks:
- Loads the fake and real news datasets.
- Prepares a validation set by combining data from both the fake and real news datasets.
- Performs feature extraction using TF-IDF (Term Frequency - Inverse Document Frequency) vectorization.
- Splits the data into training and testing sets.
- Trains a Logistic Regression model on the training set and evaluates its performance on the test set using a classification report.

### 2. `main.py`
This script combines multiple components for fake news classification:
- Merges fake and real news datasets.
- Performs manual data validation by removing certain rows from the datasets.
- Prepares the data for training by removing unnecessary columns.
- Uses TF-IDF vectorization to transform news titles into numerical features.
- Trains both Logistic Regression and Decision Tree Classifier models on the dataset.
- Provides a function for manual testing where a user can input a news article and receive predictions from both models (Logistic Regression and Decision Tree).
  
### 3. `sort.py`
This script helps filter and sort news articles into two categories:
- **Fake News**: Articles labeled as `0`.
- **Original News**: Articles labeled as `1`.
- It saves the filtered datasets into separate CSV files (`FakeNews.csv` and `OrgNews.csv`).

## Data

The project expects the following CSV files to be present in the ./data/ directory:

- FakeNews.csv: Contains news articles labeled as fake.
- TrueNews.csv: Contains news articles labeled as real.
- FakeNewsNet.csv: Another dataset for training purposes, containing news titles and labels.

## Dependencies

- pandas
- scikit-learn
- numpy
  
You can install the required dependencies using the following command:
```bash
pip install pandas scikit-learn numpy

