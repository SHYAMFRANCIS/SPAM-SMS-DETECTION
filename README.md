# SPAM-SMS-DETECTION
his repository contains a Python-based AI model designed to classify SMS messages as either spam or legitimate (ham). The project leverages various machine learning techniques for text processing, feature extraction, and classification, providing a robust solution for identifying unwanted messages.
## Features
- Data Preprocessing : Includes text cleaning, lowercasing, special character removal, tokenization, stop word removal, and stemming using NLTK.
- Flexible Data Loading : Can load SMS datasets from a specified CSV file (e.g., Kaggle datasets) or use a built-in sample dataset for demonstration.
- Feature Extraction : Supports TF-IDF (Term Frequency-Inverse Document Frequency) and Count Vectorizer for converting text data into numerical features.
- Multiple Classification Models : Implements and evaluates several popular machine learning models:
  - Naive Bayes (MultinomialNB)
  - Logistic Regression
  - Support Vector Machine (SVC)
  - Random Forest Classifier
- Model Evaluation : Provides comprehensive evaluation metrics including accuracy, AUC score, classification reports, and confusion matrices.
- Visualization : Generates plots for model accuracy comparison, ROC curves, and AUC score comparison.
- Feature Importance Analysis : Identifies key words and phrases that are most indicative of spam or ham messages (for Logistic Regression).
- Interactive Prediction : Allows users to input new SMS messages and get real-time spam/ham predictions.
 ## Getting Started
 ### Prerequisites
Ensure you have Python installed (version 3.6 or higher is recommended). You will also need the following Python libraries:

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- nltk
You can install them using pip:pip install pandas numpy matplotlib seaborn scikit-learn nltk
### Dataset
This project is designed to work with SMS spam datasets, commonly found on platforms like Kaggle. A typical dataset should contain at least two columns: one for the message content and one for the label (e.g., 'ham' or 'spam').

Example Dataset Format (e.g., spam.csv ):
### Installation and Usage
1. Clone the repository (or download the SPAM SMS DETECTION.py file):
git clone <repository_url>
cd SPAM SMS DETECTION
2. Download your Kaggle dataset (e.g., spam.csv ) and place it in the same directory as the SPAM SMS DETECTION.py script, or note its full path.
3. Update the SPAM SMS DETECTION.py file :
   
   Open SPAM SMS DETECTION.py and modify the kaggle_data_path variable in the if __name__ == "__main__": block to point to your dataset. For example:
// ... existing code ...
