# services/sentiment_analyzer.py
import logging
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# This is the VADER-based sentiment analyzer. It is a lightweight, reliable,
# and pure-Python alternative to the complex transformers library.

class SentimentAnalyzer:
    """
    Analyzes sentiment using the VADER (Valence Aware Dictionary and sEntiment
    Reasoner) model from the NLTK library. It is much lighter and more
    reliable for cross-platform compatibility than FinBERT.
    """
    def __init__(self):
        self.analyzer = None
        try:
            # VADER requires a one-time download of its lexicon.
            nltk.download('vader_lexicon')
            self.analyzer = SentimentIntensityAnalyzer()
            logging.info("VADER sentiment analyzer initialized successfully.")
        except Exception as e:
            logging.error(f"Failed to initialize VADER sentiment analyzer: {e}")
            logging.error("Please ensure you have an active internet connection for the initial download.")

    def analyze_sentiment(self, headlines):
        """
        Analyzes a list of headlines and returns a single compound sentiment score.
        Scores range from -1 (most negative) to +1 (most positive).
        """
        if not self.analyzer or not headlines:
            return 0.0

        total_score = 0
        for headline in headlines:
            # The 'compound' score is a normalized, single metric for sentiment.
            vs = self.analyzer.polarity_scores(headline)
            total_score += vs['compound']
        
        # Return the average compound score
        return total_score / len(headlines) if headlines else 0.0

