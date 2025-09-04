# services/sentiment_analyzer.py
import logging
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch


class SentimentAnalyzer:
    """
    A self-contained service for analyzing the sentiment of financial news headlines
    using the FinBERT pre-trained model.
    """

    def __init__(self):
        """
        Initializes the tokenizer and model. This can take a moment on first run
        as it may need to download the model files.
        """
        self.model_name = "ProsusAI/finbert"
        logging.info(f"Initializing SentimentAnalyzer with model: {self.model_name}...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            logging.info("FinBERT model and tokenizer loaded successfully.")
        except Exception as e:
            logging.critical(f"FATAL: Could not load FinBERT model. This is a critical error. {e}")
            # In a real application, you might want to handle this more gracefully,
            # but for now, we'll log it as critical.
            self.model = None
            self.tokenizer = None

    def analyze_sentiment(self, headlines: list[str]) -> float:
        """
        Analyzes a list of headlines and returns an aggregated sentiment score.

        The score is calculated based on the softmax probabilities of the 'positive',
        'negative', and 'neutral' classes.
        Score = (positive_prob - negative_prob)
        This results in a score between -1.0 (very negative) and +1.0 (very positive).

        Args:
            headlines: A list of string headlines.

        Returns:
            An aggregated sentiment score as a float, or 0.0 if analysis fails.
        """
        if not self.model or not self.tokenizer or not headlines:
            return 0.0

        try:
            inputs = self.tokenizer(headlines, padding=True, truncation=True, return_tensors='pt', max_length=512)

            with torch.no_grad():
                outputs = self.model(**inputs)

            # Convert logits to probabilities
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

            # The model labels are: 0 (positive), 1 (negative), 2 (neutral)
            positive_prob = predictions[:, 0].tolist()
            negative_prob = predictions[:, 1].tolist()
            # neutral_prob = predictions[:, 2].tolist() # Not used in the final score calculation

            # Calculate the score for each headline and then average them
            scores = [p - n for p, n in zip(positive_prob, negative_prob)]
            aggregated_score = sum(scores) / len(scores) if scores else 0.0

            return aggregated_score

        except Exception as e:
            logging.error(f"An error occurred during sentiment analysis: {e}")
            return 0.0

