# You will need to install the 'transformers' and 'torch' libraries first.
# Run this in your terminal: pip install transformers torch

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

class SentimentAnalyzer:
    """
    A class to analyze the sentiment of financial news headlines using the FinBERT model.
    """
    def __init__(self):
        """
        Initializes the analyzer by loading the pre-trained FinBERT model and tokenizer.
        This happens only once when the class is created.
        """
        try:
            print("Initializing Sentiment Analyzer: Loading FinBERT model...")
            self.tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
            self.model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
            print("✅ FinBERT model loaded successfully.")
        except Exception as e:
            print(f"❌ Error loading FinBERT model: {e}")
            self.model = None
            self.tokenizer = None

    def analyze_sentiment(self, headlines: list[str]) -> dict:
        """
        Analyzes a list of headlines and returns an aggregate sentiment score.

        Args:
            headlines (list[str]): A list of news headlines.

        Returns:
            dict: A dictionary containing the overall sentiment ('Positive', 'Negative', 'Neutral'),
                  the average confidence score, and the raw scores for each headline.
                  Returns {'error': 'Model not loaded'} if initialization failed.
        """
        if not self.model or not self.tokenizer:
            return {'error': 'Model not loaded'}

        if not headlines:
            return {'sentiment': 'Neutral', 'score': 0.0, 'details': []}

        try:
            inputs = self.tokenizer(headlines, padding=True, truncation=True, return_tensors='pt')
            
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            positive_scores = predictions[:, 0].tolist()
            negative_scores = predictions[:, 1].tolist()
            neutral_scores = predictions[:, 2].tolist()

            # --- Calculate Aggregate Score ---
            # We care most about the balance between positive and negative sentiment.
            avg_positive = sum(positive_scores) / len(headlines)
            avg_negative = sum(negative_scores) / len(headlines)

            final_score = avg_positive - avg_negative # A simple score from -1 (very negative) to +1 (very positive)

            if final_score > 0.1:
                sentiment = "Positive"
            elif final_score < -0.1:
                sentiment = "Negative"
            else:
                sentiment = "Neutral"

            return {
                "sentiment": sentiment,
                "score": round(final_score, 4),
                "details": {
                    "avg_positive": round(avg_positive, 4),
                    "avg_negative": round(avg_negative, 4)
                }
            }

        except Exception as e:
            return {'error': f'An error occurred during analysis: {e}'}


# --- Example Usage ---
if __name__ == '__main__':
    # This block will only run when you execute this file directly (e.g., "python sentiment_analyzer.py")
    
    analyzer = SentimentAnalyzer()

    if analyzer.model:
        # --- Test Case 1: Positive News ---
        positive_headlines = [
            "Stock surges after record earnings report",
            "Company announces breakthrough technology and raises guidance",
            "Analyst upgrades rating to 'Strong Buy'"
        ]
        positive_result = analyzer.analyze_sentiment(positive_headlines)
        print(f"\nPositive Headlines Test Result: {positive_result}")

        # --- Test Case 2: Negative News ---
        negative_headlines = [
            "CEO resigns amid SEC investigation",
            "Company misses earnings expectations and lowers forecast",
            "Factory shutdown expected to impact Q4 results"
        ]
        negative_result = analyzer.analyze_sentiment(negative_headlines)
        print(f"Negative Headlines Test Result: {negative_result}")

        # --- Test Case 3: Ambiguous/Neutral News ---
        neutral_headlines = [
            "Market awaits Federal Reserve interest rate decision",
            "Company to present at upcoming industry conference"
        ]
        neutral_result = analyzer.analyze_sentiment(neutral_headlines)
        print(f"Neutral Headlines Test Result: {neutral_result}")
