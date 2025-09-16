import logging
import nltk
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    """
    A specialist module for performing sentiment analysis on news headlines.
    Uses the VADER (Valence Aware Dictionary and sEntiment Reasoner) model,
    which is a lightweight, pure-Python engine well-suited for our needs.
    This avoids the heavy dependencies and "posix Ghost" issues of past versions.
    """
    def __init__(self, config):
        """
        Initializes the analyzer and ensures the necessary NLTK data is available.
        Args:
            config: The main configuration object.
        """
        self.config = config
        self.analyzer = SentimentIntensityAnalyzer()
        # This is a one-time check and download. If the data exists, it does nothing.
        # This prevents errors if the bot is run in a fresh environment.
        self._ensure_vader_lexicon_is_downloaded()

    def _ensure_vader_lexicon_is_downloaded(self):
        """
        A private helper method to check for and download the VADER lexicon
        if it's not already present.
        """
        try:
            # A fast way to check if the data package is available.
            nltk.data.find('sentiment/vader_lexicon.zip')
        except LookupError:
            logger.info("VADER lexicon not found. Downloading...")
            # nltk.download() is a blocking operation, but it's only called once
            # during startup, which is an acceptable trade-off for simplicity.
            nltk.download('vader_lexicon')
            logger.info("VADER lexicon downloaded successfully.")

    def get_sentiment(self, text: str) -> float:
        """
        Analyzes a string of text and returns its compound sentiment score.
        Args:
            text (str): The text to be analyzed (e.g., a news headline).
        Returns:
            A float between -1 (most negative) and 1 (most positive).
            The 'compound' score is a normalized, weighted composite score.
        """
        if not isinstance(text, str):
            logger.warning("Sentiment analysis received non-string input. Returning neutral score.")
            return 0.0
            
        # The polarity_scores() method returns a dictionary with neg, neu, pos,
        # and compound scores. We are primarily interested in the compound score.
        sentiment_scores = self.analyzer.polarity_scores(text)
        compound_score = sentiment_scores['compound']
        
        logger.debug(f"Sentiment for '{text[:50]}...': {compound_score}")
        return compound_score
