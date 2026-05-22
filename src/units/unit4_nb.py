import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report
import re

# Ensure NLTK resources are available
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

class OutbreakTextDetector:
    """Unit IV: Naive Bayes for Citizen Complaint Classification."""
    def __init__(self, model_type='multinomial'):
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        
        if model_type == 'multinomial':
            self.model = MultinomialNB(alpha=1.0) # Laplace Smoothing
        else:
            self.model = BernoulliNB(alpha=1.0)
            
        self.pipeline = Pipeline([
            ('vectorizer', CountVectorizer(preprocessor=self.preprocess_text)),
            ('nb', self.model)
        ])

    def preprocess_text(self, text):
        """Standard NLP preprocessing: cleaning, stopword removal, and stemming."""
        text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
        words = text.split()
        words = [self.stemmer.stem(w) for w in words if w not in self.stop_words]
        return " ".join(words)

    def train(self, texts, labels):
        """Trains the NB classifier and returns test metrics."""
        X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)
        self.pipeline.fit(X_train, y_train)
        
        y_pred = self.pipeline.predict(X_test)
        return classification_report(y_test, y_pred)

    def predict(self, texts):
        return self.pipeline.predict(texts)

    def get_feature_importance(self):
        """Returns words most indicative of contamination."""
        nb = self.pipeline.named_steps['nb']
        vectorizer = self.pipeline.named_steps['vectorizer']
        
        if hasattr(nb, 'feature_log_prob_'):
            feature_probs = nb.feature_log_prob_[1]
            words = vectorizer.get_feature_names_out()
            importance_df = pd.DataFrame({'Word': words, 'LogProb': feature_probs})
            return importance_df.sort_values(by='LogProb', ascending=False).head(20)
        return pd.DataFrame()
