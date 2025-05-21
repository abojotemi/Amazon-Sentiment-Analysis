import streamlit as st
import pandas as pd
import numpy as np
import re
import pickle
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import time
import matplotlib.pyplot as plt

# Set page configuration
st.set_page_config(
    page_title="Sentiment Analysis",
    page_icon="😊",
    layout="wide"
)

# Download NLTK resources
@st.cache_resource
def download_nltk_resources():
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

download_nltk_resources()

# Load models
@st.cache_resource
def load_models():
    try:
        # Load the trained models
        models = {
            'Random Forest': pickle.load(open('models/random_forest_model.pkl', 'rb')),
            'Gradient Boosting': pickle.load(open('models/gradient_boosting_model.pkl', 'rb')),
            'XGBoost': pickle.load(open('models/xgboost_model.pkl', 'rb')),
            'CatBoost': pickle.load(open('models/catboost_model.pkl', 'rb')),
            'AdaBoost': pickle.load(open('models/adaboost_model.pkl', 'rb')),
            'Bagging': pickle.load(open('models/bagging_model.pkl', 'rb'))
        }
        # Load the TF-IDF vectorizer
        tfidf_vectorizer = pickle.load(open('models/tfidf_vectorizer.pkl', 'rb'))
        # Load the label encoder
        label_encoder = pickle.load(open('models/label_encoder.pkl', 'rb'))
        
        return models, tfidf_vectorizer, label_encoder
    except Exception as e:
        st.error(f"Error loading models: {e}")
        # For demonstration, we'll create dummy models
        return create_dummy_models()

def create_dummy_models():
    """Create dummy models for demonstration purposes"""
    st.warning("Using dummy models for demonstration. Predictions will not be accurate.")
    
    # Dummy models for demonstration
    base_clf = DecisionTreeClassifier(max_depth=5, random_state=42)
    
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=10, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=10, random_state=42),
        'XGBoost': XGBClassifier(n_estimators=10, random_state=42, eval_metric='logloss'),
        'CatBoost': CatBoostClassifier(iterations=10, random_state=42, verbose=False),
        'AdaBoost': AdaBoostClassifier(estimator=base_clf, n_estimators=10, random_state=42),
        'Bagging': BaggingClassifier(estimator=base_clf, n_estimators=10, random_state=42)
    }
    
    # Create a simple vectorizer and encoder
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import LabelEncoder
    
    tfidf_vectorizer = TfidfVectorizer(max_features=1000)
    label_encoder = LabelEncoder()
    label_encoder.classes_ = np.array(['negative', 'positive'])
    
    return models, tfidf_vectorizer, label_encoder

# Text preprocessing function
def preprocess_text(text):
    """Apply all preprocessing steps"""
    if not isinstance(text, str):
        return ""

    # Normalization - Convert to lowercase
    text = text.lower()

    # Remove special characters and numbers
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)

    # Tokenization
    tokens = word_tokenize(text)

    # Stop word removal
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]

    # Join tokens back into text
    processed_text = ' '.join(tokens)

    return processed_text

# Predict sentiment function
def predict_sentiment(text, model_name, models, vectorizer, encoder):
    # Preprocess the text
    processed_text = preprocess_text(text)
    
    # Vectorize the text
    text_vectorized = vectorizer.transform([processed_text])
    
    # Make prediction
    start_time = time.time()
    model = models[model_name]
    
    try:
        # Get prediction probabilities if the model supports it
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(text_vectorized)[0]
            prediction_idx = np.argmax(probabilities)
            confidence = probabilities[prediction_idx]
            prediction = encoder.classes_[prediction_idx]
        else:
            # For models that don't support predict_proba
            prediction_idx = model.predict(text_vectorized)[0]
            prediction = encoder.classes_[prediction_idx]
            confidence = 1.0  # Default confidence
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return "Error", 0.0, 0.0
        
    prediction_time = time.time() - start_time
    
    return prediction, confidence, prediction_time

# Function to create the sentiment gauge chart
def create_sentiment_gauge(confidence, sentiment):
    fig, ax = plt.subplots(figsize=(4, 3))
    
    # Determine the position based on sentiment and confidence
    if sentiment == 'positive':
        position = 0.5 + (confidence / 2)
        color = 'green'
    else:
        position = 0.5 - (confidence / 2)
        color = 'red'
    
    # Create a gauge-like visualization
    ax.barh(0, 1, height=0.5, color='lightgray')
    ax.barh(0, position, height=0.5, color=color)
    
    # Add an indicator line
    ax.plot([0.5, 0.5], [-0.5, 0.5], 'k--', alpha=0.5)
    
    # Add labels
    ax.text(0.05, 0, 'Negative', ha='left', va='center')
    ax.text(0.95, 0, 'Positive', ha='right', va='center')
    
    # Clean up the chart
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.5, 0.5)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    
    return fig

# Main app function
def main():
    # Load models, vectorizer and encoder
    models, tfidf_vectorizer, label_encoder = load_models()
    
    # Create the app header
    st.title("📊 Sentiment Analysis App")
    st.subheader("Analyze the sentiment of your text")
    
    # Create two columns for the app layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Text input area
        text_input = st.text_area("Enter your review or text:", 
                                 height=150,
                                 placeholder="Type your review here... (e.g., I really enjoyed this product, it exceeded my expectations!)")
        
        # Model selection
        model_name = st.selectbox("Select Model:", 
                                 list(models.keys()),
                                 index=0)
        
        # Add information about the selected model
        model_info = {
            'Random Forest': "A versatile ensemble learning method that operates by constructing multiple decision trees during training.",
            'Gradient Boosting': "Builds an ensemble of trees one at a time, where each new tree helps to correct errors made by previously trained trees.",
            'XGBoost': "An optimized distributed gradient boosting library designed to be highly efficient, flexible and portable.",
            'CatBoost': "A gradient boosting algorithm that handles categorical features automatically and produces better results with default parameters.",
            'AdaBoost': "Focuses on classification problems by combining multiple 'weak classifiers' into a single 'strong classifier'.",
            'Bagging': "Builds multiple estimators independently from random subsets of the training data."
        }
        
        st.caption(f"**Model Description:** {model_info[model_name]}")
        
        # Analyze button
        if st.button("Analyze Sentiment", type="primary"):
            if text_input.strip() == "":
                st.error("Please enter some text to analyze.")
            else:
                with st.spinner("Analyzing sentiment..."):
                    # Make prediction
                    prediction, confidence, prediction_time = predict_sentiment(
                        text_input, model_name, models, tfidf_vectorizer, label_encoder
                    )
                    
                    # Display results
                    st.markdown("### Results")
                    
                    # Show sentiment with appropriate styling
                    if prediction == "positive":
                        st.markdown(f"<h3 style='color: green;'>Sentiment: {prediction.capitalize()} 😊</h3>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"<h3 style='color: red;'>Sentiment: {prediction.capitalize()} 😞</h3>", unsafe_allow_html=True)
                    
                    # Show confidence
                    st.markdown(f"**Confidence:** {confidence:.2%}")
                    
                    # Show processing time
                    st.markdown(f"**Processing Time:** {prediction_time:.4f} seconds")
                    
    with col2:
        # Show the model info section
        st.markdown("### Model Information")
        st.markdown("""
        This app uses ensemble machine learning models trained on Amazon product reviews to predict sentiment.
        
        **Models Available:**
        - Random Forest
        - Gradient Boosting
        - XGBoost
        - CatBoost
        - AdaBoost
        - Bagging
        
        **How It Works:**
        1. Enter your review text
        2. Select a model
        3. Click "Analyze Sentiment"
        4. Get prediction results
        """)
        
        # Display sentiment gauge when there's a prediction
        if 'prediction' in locals():
            st.markdown("### Sentiment Meter")
            st.pyplot(create_sentiment_gauge(confidence, prediction))
        
    # Add a section for processing steps
    st.markdown("---")
    with st.expander("How text preprocessing works"):
        st.markdown("""
        Your text goes through these steps before analysis:
        
        1. **Normalization** - Convert to lowercase
        2. **Cleaning** - Remove special characters and numbers
        3. **Tokenization** - Split text into individual words
        4. **Stop Word Removal** - Remove common words like 'the', 'and', etc.
        5. **Lemmatization** - Convert words to their base form
        6. **Vectorization** - Convert text to numerical features using TF-IDF
        7. **Prediction** - Apply the selected model to predict sentiment
        """)
        
        # Show a sample of processed text if input is provided
        if 'text_input' in locals() and text_input.strip() != "":
            st.markdown("### Sample Processing")
            processed_text = preprocess_text(text_input)
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Original Text:**")
                st.markdown(f"```\n{text_input}\n```")
            with col2:
                st.markdown("**Processed Text:**")
                st.markdown(f"```\n{processed_text}\n```")

    # Footer
    st.markdown("---")
    st.markdown("Built with ❤️ using Streamlit and ensemble ML models")

if __name__ == "__main__":
    main()