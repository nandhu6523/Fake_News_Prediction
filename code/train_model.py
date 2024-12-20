import pandas as pd
import re
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pickle

# 1. Data Processing
def load_data(file_path):
    df = pd.read_csv(file_path)
    print("Data loaded successfully!")
    return df

def preprocess_text(text):
    """Preprocess the input text by removing URLs, mentions, and special characters."""
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)  # Remove URLs
    text = re.sub(r'\@\w+|\#', '', text)  # Remove mentions and hashtags
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove punctuation and numbers
    return text.lower()

def process_data(df):
    """Process the dataset: handle missing values and preprocess text."""
    # Check for missing values
    print("Missing values before processing:\n", df.isnull().sum())
    df.dropna(inplace=True)  # Drop rows with missing values
    df['text'] = df['text'].apply(preprocess_text)
    print("Data processed successfully!")
    return df

# 2. Feature Extraction
def visualize_class_distribution(df):
    """Visualize the distribution of classes in the dataset."""
    df['label'].value_counts().plot(kind='bar')
    plt.title('Class Distribution')
    plt.xlabel('Label')
    plt.ylabel('Count')
    plt.show()

def extract_features(X_train, X_test):
    """Vectorize the text data using TF-IDF."""
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), max_df=0.7)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    return X_train_tfidf, X_test_tfidf, vectorizer

# 3. Classification
def train_model(X_train_tfidf, y_train):
    """Train the Logistic Regression model."""
    model = LogisticRegression()
    model.fit(X_train_tfidf, y_train)
    return model

def evaluate_model(model, X_test_tfidf, y_test):
    """Evaluate the model's performance on the test set."""
    y_pred = model.predict(X_test_tfidf)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

def save_model(model, vectorizer):
    """Save the trained model and vectorizer to files."""
    with open('model.pkl', 'wb') as model_file:
        pickle.dump(model, model_file)
    with open('vectorizer.pkl', 'wb') as vec_file:
        pickle.dump(vectorizer, vec_file)
    print("Model and vectorizer saved successfully!")

# Main Execution
if __name__ == "__main__":
    # Load and process the dataset
    df = load_data('news1.csv')
    df = process_data(df)

    # Visualize class distribution
    visualize_class_distribution(df)

    # Encode labels to numeric
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df['label'])  # Convert to 1 for REAL, 0 for FAKE

    # Split the data into features (X) and labels (y)
    X = df['text']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Extract features
    X_train_tfidf, X_test_tfidf, vectorizer = extract_features(X_train, X_test)

    # Train the model
    model = train_model(X_train_tfidf, y_train)

    # Evaluate the model
    evaluate_model(model, X_test_tfidf, y_test)

    # Save the model and vectorizer
    save_model(model, vectorizer)
