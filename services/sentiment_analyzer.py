"""
Services/sentiment_analyzer.py

Author: 403-Forbidden
Purpose: Provides sentiment analysis services for raw text messages using the
         Vader (Valence Aware Dictionary and sEntiment Reasoner) model.
"""
import logging
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

class SentimentAnalyzer:
    """
    A wrapper for the Vader sentiment analysis tool.
    """
    def __init__(self):
        """
        Initializes the SentimentIntensityAnalyzer.
        FIX: Enhanced error handling with startup warning.
        """
        try:
            self.analyzer = SentimentIntensityAnalyzer()
            logging.info("VADER sentiment analyzer initialized successfully.")
        except Exception as e:
            logging.critical(f"FAILED to initialize SentimentIntensityAnalyzer: {e}")
            logging.critical("Sentiment filter will NOT work. Bot will proceed without sentiment veto.")
            self.analyzer = None

    def get_sentiment_score(self, text: str) -> float:
        """
        Analyzes a string of text and returns its compound sentiment score.

        The compound score is a metric that calculates the sum of all lexicon ratings
        which have been normalized between -1 (most extreme negative) and +1
        (most extreme positive).

        Args:
            text (str): The raw text of the message to analyze.

        Returns:
            float: The compound sentiment score. Returns None if analyzer failed to initialize.
        """
        if not self.analyzer:
            logging.warning("Sentiment analyzer is not available. Returning None.")
            return None
            
        try:
            # The polarity_scores() method returns a dict with 'neg', 'neu', 'pos', 'compound'
            sentiment_scores = self.analyzer.polarity_scores(text)
            
            # The 'compound' score is the most useful for a single-metric threshold.
            compound_score = sentiment_scores['compound']
            
            logging.debug(f"Analyzed sentiment for text: '{text[:50]}...'. Compound Score: {compound_score}")
            
            return compound_score
        except Exception as e:
            logging.error(f"Error during sentiment analysis: {e}", exc_info=True)
            return None
