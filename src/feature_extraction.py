import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np


class FeatureExtractor:
    def __init__(self):
        try:
            self.nlp = spacy.load('en_core_web_sm')
        except:
            print("Warning: spaCy model not loaded. Named entity extraction will be limited.")
            self.nlp = None

        # OPTIMIZED TF-IDF settings for resume classification
        self.tfidf = TfidfVectorizer(
            max_features=2000,  # Increased features for better representation
            ngram_range=(1, 3),  # Uni-grams, bi-grams, and tri-grams
            min_df=2,  # Ignore terms appearing in less than 2 documents
            max_df=0.85,  # Ignore terms appearing in more than 85% of documents
            sublinear_tf=True,  # Use logarithmic term frequency
            strip_accents='unicode',
            lowercase=True,
            # Custom token pattern to preserve alphanumeric + common tech terms
            token_pattern=r'\b[a-zA-Z][a-zA-Z0-9]+\b'
        )

    def extract_named_entities(self, text):
        """Extract named entities using spaCy"""
        if not self.nlp:
            return {}

        doc = self.nlp(text)
        entities = {
            'PERSON': [],
            'ORG': [],
            'GPE': [],
            'DATE': [],
            'SKILL': []  # Custom category for skills
        }

        for ent in doc.ents:
            if ent.label_ in entities:
                entities[ent.label_].append(ent.text)

        return entities

    def fit_tfidf(self, texts):
        """Fit TF-IDF vectorizer on training data"""
        return self.tfidf.fit_transform(texts)

    def transform_tfidf(self, texts):
        """Transform text to TF-IDF features"""
        return self.tfidf.transform(texts)

    def get_feature_names(self):
        """Get feature names from TF-IDF vectorizer"""
        return self.tfidf.get_feature_names_out()