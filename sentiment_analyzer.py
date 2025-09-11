# services/sentiment_analyzer.py
import logging
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

class SentimentAnalyzer:
    """
    Analyzes sentiment using the VADER model. This version is hardened
    against ZeroDivisionError if an empty list of headlines is provided.
    """
    def __init__(self):
        self.analyzer = None
        try:
            nltk.download('vader_lexicon')
            self.analyzer = SentimentIntensityAnalyzer()
            logging.info("VADER sentiment analyzer initialized successfully.")
        except Exception as e:
            logging.error(f"Failed to initialize VADER sentiment analyzer: {e}")

    def analyze_sentiment(self, headlines):
        """
        Analyzes a list of headlines and returns a single compound sentiment score.
        """
        if not self.analyzer or not headlines:
            # --- THIS IS THE CRITICAL FIX ---
            # If the list is empty, return a neutral score immediately.
            return 0.0

        total_score = 0
        for headline in headlines:
            vs = self.analyzer.polarity_scores(headline)
            total_score += vs['compound']
        
        # This division is now safe because we've already checked if headlines is empty.
        return total_score / len(headlines)

