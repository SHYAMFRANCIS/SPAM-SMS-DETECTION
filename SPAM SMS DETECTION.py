import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import warnings

warnings.filterwarnings('ignore')

class SMSSpamDetector:
    def __init__(self, data_path=None):
        self.vectorizer = None
        self.models = {}
        self.model_scores = {}
        self.best_model = None
        self.stemmer = PorterStemmer()
        self.data_path = data_path # New attribute to store data path
        
        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('punkt')
            nltk.download('stopwords')
        
        self.stop_words = set(stopwords.words('english'))
    
    def load_data(self):
        """Load SMS data from a CSV file or create sample data if no path is provided"""
        if self.data_path:
            try:
                # Assuming the Kaggle dataset is a CSV with 'v1' for label and 'v2' for message
                df = pd.read_csv(self.data_path, encoding='latin-1')
                df = df.iloc[:, :2] # Take only the first two columns
                df.columns = ['label', 'message'] # Rename columns
                # Map labels to 'ham' and 'spam' if they are 'v1' and 'v2' or similar
                df['label'] = df['label'].map({'ham': 'ham', 'spam': 'spam'}) # Ensure consistent labels
                df.dropna(inplace=True)
                print(f"Successfully loaded data from {self.data_path}")
                return df
            except FileNotFoundError:
                print(f"Error: Dataset not found at {self.data_path}. Creating sample data instead.")
                return self._create_sample_data()
            except Exception as e:
                print(f"Error loading data from {self.data_path}: {e}. Creating sample data instead.")
                return self._create_sample_data()
        else:
            print("No data path provided. Creating sample data for demonstration.")
            return self._create_sample_data()

    def _create_sample_data(self):
        """Create sample SMS data for demonstration (internal helper)"""
        spam_messages = [
            "URGENT! You have won a $1000 cash prize! Call now to claim your reward!",
            "Congratulations! You've been selected for a FREE iPhone! Click here now!",
            "WINNER! You've won £500 cash! Reply CLAIM to collect your prize money!",
            "Limited time offer! Get 90% discount on all products! Buy now!",
            "Your loan has been approved for $50000! No credit check required!",
            "Act now! Free gift cards available! Click the link to claim yours!",
            "URGENT: Your account will be suspended! Click link to verify now!",
            "You have 2 missed calls from unknown number. Call back to win prize!",
            "HOT SINGLES in your area want to meet you! Click here now!",
            "Get rich quick! Make $5000 per week working from home!",
            "FREE RINGTONES! Text STOP to cancel. Standard rates apply.",
            "Your mobile number has won 1 MILLION dollars! Call now!",
            "BANK ALERT: Confirm your account by clicking this link immediately!",
            "Final notice: Your subscription will auto-renew. Reply STOP to cancel.",
            "You have qualified for a 0% interest credit card! Apply now!"
        ]
        
        legitimate_messages = [
            "Hey, are you free for dinner tonight? Let me know!",
            "Thanks for the birthday wishes! Had a great time at the party.",
            "Meeting moved to 3 PM tomorrow. See you in conference room B.",
            "Can you pick up milk on your way home? We're running low.",
            "The project deadline has been extended to next Friday.",
            "Your package has been delivered. Thanks for shopping with us!",
            "Reminder: Doctor appointment tomorrow at 2 PM.",
            "Great job on the presentation today! Well done.",
            "Traffic is heavy on Route 95. Consider taking alternate route.",
            "Your flight has been delayed by 30 minutes. New boarding time: 6:30 PM",
            "Happy anniversary! Hope you have a wonderful celebration.",
            "The weather looks good for our picnic this weekend.",
            "Don't forget to bring your laptop to tomorrow's meeting.",
            "Your library books are due next Tuesday. Please return them.",
            "Thanks for helping me move. Pizza is on me next weekend!"
        ]
        
        # Create DataFrame
        messages = spam_messages + legitimate_messages
        labels = ['spam'] * len(spam_messages) + ['ham'] * len(legitimate_messages)
        
        return pd.DataFrame({
            'message': messages,
            'label': labels
        })
    
    def preprocess_text(self, text):
        """Preprocess text data"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and apply stemming
        tokens = [self.stemmer.stem(token) for token in tokens if token not in self.stop_words]
        
        return ' '.join(tokens)
    
    def prepare_data(self, df):
        """Prepare data for training"""
        # Preprocess messages
        df['processed_message'] = df['message'].apply(self.preprocess_text)
        
        # Convert labels to binary
        df['label_binary'] = df['label'].map({'ham': 0, 'spam': 1})
        
        return df
    
    def extract_features(self, X_train, X_test, method='tfidf'):
        """Extract features using TF-IDF or Count Vectorizer"""
        if method == 'tfidf':
            self.vectorizer = TfidfVectorizer(
                max_features=5000,
                ngram_range=(1, 2),
                stop_words='english'
            )
        else:
            self.vectorizer = CountVectorizer(
                max_features=5000,
                ngram_range=(1, 2),
                stop_words='english'
            )
        
        X_train_features = self.vectorizer.fit_transform(X_train)
        X_test_features = self.vectorizer.transform(X_test)
        
        return X_train_features, X_test_features
    
    def train_models(self, X_train, y_train):
        """Train multiple classification models"""
        models = {
            'Naive Bayes': MultinomialNB(),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'SVM': SVC(random_state=42, probability=True),
            'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100)
        }
        
        for name, model in models.items():
            print(f"Training {name}...")
            model.fit(X_train, y_train)
            self.models[name] = model
            
            # Cross-validation scores
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
            self.model_scores[name] = {
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std()
            }
            print(f"{name} - CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    def evaluate_models(self, X_test, y_test):
        """Evaluate all trained models"""
        results = {}
        
        for name, model in self.models.items():
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            accuracy = accuracy_score(y_test, y_pred)
            auc_score = roc_auc_score(y_test, y_pred_proba)
            
            results[name] = {
                'accuracy': accuracy,
                'auc': auc_score,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba
            }
            
            print(f"\n{name} Results:")
            print(f"Accuracy: {accuracy:.4f}")
            print(f"AUC Score: {auc_score:.4f}")
            print("\nClassification Report:")
            print(classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))
        
        return results
    
    def plot_results(self, results, y_test):
        """Plot evaluation results"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Model Accuracy Comparison
        models = list(results.keys())
        accuracies = [results[model]['accuracy'] for model in models]
        
        axes[0, 0].bar(models, accuracies, color=['skyblue', 'lightgreen', 'lightcoral', 'gold'])
        axes[0, 0].set_title('Model Accuracy Comparison')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].set_ylim(0, 1)
        for i, v in enumerate(accuracies):
            axes[0, 0].text(i, v + 0.01, f'{v:.3f}', ha='center')
        
        # 2. ROC Curves
        for model_name, result in results.items():
            fpr, tpr, _ = roc_curve(y_test, result['y_pred_proba'])
            axes[0, 1].plot(fpr, tpr, label=f"{model_name} (AUC = {result['auc']:.3f})")
        
        axes[0, 1].plot([0, 1], [0, 1], 'k--', label='Random')
        axes[0, 1].set_xlabel('False Positive Rate')
        axes[0, 1].set_ylabel('True Positive Rate')
        axes[0, 1].set_title('ROC Curves')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # 3. Confusion Matrix for best model
        best_model_name = max(results.keys(), key=lambda x: results[x]['accuracy'])
        best_result = results[best_model_name]
        
        cm = confusion_matrix(y_test, best_result['y_pred'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'], ax=axes[1, 0])
        axes[1, 0].set_title(f'Confusion Matrix - {best_model_name}')
        axes[1, 0].set_ylabel('True Label')
        axes[1, 0].set_xlabel('Predicted Label')
        
        # 4. AUC Score Comparison
        auc_scores = [results[model]['auc'] for model in models]
        
        axes[1, 1].bar(models, auc_scores, color=['skyblue', 'lightgreen', 'lightcoral', 'gold'])
        axes[1, 1].set_title('Model AUC Score Comparison')
        axes[1, 1].set_ylabel('AUC Score')
        axes[1, 1].set_ylim(0, 1)
        for i, v in enumerate(auc_scores):
            axes[1, 1].text(i, v + 0.01, f'{v:.3f}', ha='center')
        
        plt.tight_layout()
        plt.show()
        
        return best_model_name
    
    def predict_message(self, message, model_name=None):
        """Predict if a single message is spam or ham"""
        if model_name is None:
            model_name = self.best_model
        
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found. Available models: {list(self.models.keys())}")
        
        # Preprocess the message
        processed_message = self.preprocess_text(message)
        
        # Transform using the fitted vectorizer
        message_features = self.vectorizer.transform([processed_message])
        
        # Get prediction and probability
        prediction = self.models[model_name].predict(message_features)[0]
        probability = self.models[model_name].predict_proba(message_features)[0]
        
        result = {
            'message': message,
            'prediction': 'spam' if prediction == 1 else 'ham',
            'confidence': max(probability),
            'spam_probability': probability[1],
            'ham_probability': probability[0]
        }
        
        return result
    
    def feature_importance_analysis(self):
        """Analyze feature importance for interpretability"""
        if 'Logistic Regression' in self.models:
            model = self.models['Logistic Regression']
            feature_names = self.vectorizer.get_feature_names_out()
            
            # Get coefficients (importance scores)
            coefficients = model.coef_[0]
            
            # Get top spam-indicating features
            spam_features_idx = np.argsort(coefficients)[-20:]
            spam_features = [(feature_names[i], coefficients[i]) for i in spam_features_idx]
            
            # Get top ham-indicating features
            ham_features_idx = np.argsort(coefficients)[:20]
            ham_features = [(feature_names[i], coefficients[i]) for i in ham_features_idx]
            
            print("Top 20 Spam-indicating features:")
            for feature, coef in reversed(spam_features):
                print(f"{feature}: {coef:.4f}")
            
            print("\nTop 20 Ham-indicating features:")
            for feature, coef in ham_features:
                print(f"{feature}: {coef:.4f}")
            
            return spam_features, ham_features
    
    def run_complete_analysis(self):
        """Run the complete spam detection pipeline"""
        print("=== SMS Spam Detection System ===\n")
        
        # 1. Load and prepare data
        print("1. Loading and preparing data...")
        df = self.load_data() # Call the new load_data method
        df = self.prepare_data(df)
        
        print(f"Dataset info:")
        print(f"Total messages: {len(df)}")
        print(f"Spam messages: {sum(df['label'] == 'spam')}")
        print(f"Ham messages: {sum(df['label'] == 'ham')}")
        
        # 2. Split data
        X = df['processed_message']
        y = df['label_binary']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
        
        # 3. Extract features
        print("\n2. Extracting features using TF-IDF...")
        X_train_features, X_test_features = self.extract_features(X_train, X_test)
        print(f"Feature matrix shape: {X_train_features.shape}")
        
        # 4. Train models
        print("\n3. Training models...")
        self.train_models(X_train_features, y_train)
        
        # 5. Evaluate models
        print("\n4. Evaluating models...")
        results = self.evaluate_models(X_test_features, y_test)
        
        # 6. Plot results and find best model
        print("\n5. Plotting results...")
        self.best_model = self.plot_results(results, y_test)
        print(f"\nBest performing model: {self.best_model}")
        
        # 7. Feature importance analysis
        print("\n6. Feature importance analysis...")
        self.feature_importance_analysis()
        
        # 8. Test with new messages
        print("\n7. Testing with new messages...")
        test_messages = [
            "Congratulations! You've won $1000! Call now!",
            "Hey, can you pick me up at 7 PM?",
            "URGENT! Your account needs verification!",
            "Thanks for lunch today, had a great time!"
        ]
        
        for msg in test_messages:
            result = self.predict_message(msg)
            print(f"\nMessage: '{result['message']}'")
            print(f"Prediction: {result['prediction'].upper()}")
            print(f"Confidence: {result['confidence']:.4f}")
            print(f"Spam probability: {result['spam_probability']:.4f}")

# Example usage
if __name__ == "__main__":
    # Specify the path to your Kaggle dataset here
    # For example, if your dataset is named 'spam.csv' and is in the same directory:
    kaggle_data_path = "C:\\Users\\franc\\Downloads\\archive (1)\\spam.csv"

    # Initialize and run the spam detector
    detector = SMSSpamDetector(data_path=kaggle_data_path)
    detector.run_complete_analysis()
    
    # Interactive testing
    print("\n" + "="*50)
    print("Interactive Testing")
    print("="*50)
    
    while True:
        user_input = input("\nEnter an SMS message to classify (or 'quit' to exit): ")
        if user_input.lower() == 'quit':
            break
        
        try:
            result = detector.predict_message(user_input)
            print(f"\nPrediction: {result['prediction'].upper()}")
            print(f"Confidence: {result['confidence']:.4f}")
            print(f"Spam probability: {result['spam_probability']:.4f}")
        except Exception as e:
            print(f"Error: {e}")