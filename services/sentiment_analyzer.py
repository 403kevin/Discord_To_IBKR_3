import logging
import nltk
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

logger = logging.getLogger(__name__)


class SentimentAnalyzer:
    """
    A specialist module for performing sentiment analysis on news headlines.
    This is the corrected version with the proper method name.
    """

    def __init__(self, config):
        self.config = config
        # This ensures the necessary NLTK data is downloaded once if needed.
        try:
            nltk.data.find('sentiment/vader_lexicon.zip')
        except LookupError:
            logger.info("VADER lexicon not found. Downloading...")
            nltk.download('vader_lexicon')

        self.analyzer = SentimentIntensityAnalyzer()

    def analyze_sentiment(self, text: str) -> float:
        """
        Analyzes the sentiment of a given text and returns the compound score.
        This is the corrected method name that the SignalProcessor expects.

        Args:
            text (str): The text to analyze (e.g., a news headline or message).

        Returns:
            float: A sentiment score from -1 (most negative) to 1 (most positive).
        """
        # The VADER sentiment analysis tool returns a dictionary of scores.
        # The 'compound' score is a single, normalized value that is easiest to work with.
        sentiment_scores = self.analyzer.polarity_scores(text)
        compound_score = sentiment_scores['compound']

        logger.debug(f"Sentiment analysis for '{text[:30]}...': {compound_score}")

        return compound_score

