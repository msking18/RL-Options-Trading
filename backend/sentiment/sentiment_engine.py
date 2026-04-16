from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import logging

# Configure logging
logger = logging.getLogger(__name__)

class SentimentEngine:
    """
    Analyzes sentiment of financial news headlines using VADER.
    VADER is chosen for its speed and effectiveness on short texts like headlines.
    Supports weighted aggregation: featured/top headlines carry more influence.
    """
    
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()
        
        # Add financial-specific lexicon boosts
        financial_lexicon = {
            'bullish': 2.5, 'bearish': -2.5,
            'rally': 2.0, 'surge': 2.0, 'soar': 2.0,
            'crash': -3.0, 'plunge': -2.5, 'tumble': -2.0,
            'correction': -1.5, 'slump': -2.0,
            'upgrade': 1.5, 'downgrade': -1.5,
            'outperform': 1.5, 'underperform': -1.5,
            'hawkish': -1.0, 'dovish': 1.0,
            'recession': -2.5, 'recovery': 1.5,
            'default': -3.0, 'bailout': -1.5,
        }
        self.analyzer.lexicon.update(financial_lexicon)

    def analyze_headline(self, text: str):
        """
        Calculates sentiment scores for a single headline.
        Returns a dictionary with compound, pos, neg, and neu scores.
        """
        if not text:
            return {'compound': 0, 'pos': 0, 'neg': 0, 'neu': 0}
            
        scores = self.analyzer.polarity_scores(text)
        return scores

    def aggregate_sentiment(self, headlines_data: list):
        """
        Calculates the weighted average sentiment score for a list of headlines.
        headlines_data: List of dicts with 'headline' key and optional 'weight' key.
        Featured/top headlines (weight=1.5) carry more influence than regular (weight=1.0).
        """
        if not headlines_data:
            return {
                'avg_compound': 0,
                'avg_pos': 0,
                'avg_neg': 0,
                'avg_neu': 0,
                'count': 0
            }
            
        total_compound = 0
        total_pos = 0
        total_neg = 0
        total_neu = 0
        total_weight = 0
        
        for item in headlines_data:
            scores = self.analyze_headline(item['headline'])
            w = item.get('weight', 1.0)
            total_compound += scores['compound'] * w
            total_pos += scores['pos'] * w
            total_neg += scores['neg'] * w
            total_neu += scores['neu'] * w
            total_weight += w
        
        if total_weight == 0:
            total_weight = 1  # safety
            
        return {
            'avg_compound': total_compound / total_weight,
            'avg_pos': total_pos / total_weight,
            'avg_neg': total_neg / total_weight,
            'avg_neu': total_neu / total_weight,
            'count': len(headlines_data)
        }

if __name__ == "__main__":
    # Quick test
    engine = SentimentEngine()
    test_headlines = [
        {'headline': "Nifty 50 surges to all-time high as bank stocks rally.", 'weight': 1.5},
        {'headline': "Global markets tumble on inflation fears and rate hike concerns.", 'weight': 1.5},
        {'headline': "Company X reports record-breaking quarterly profits.", 'weight': 1.0}
    ]
    
    result = engine.aggregate_sentiment(test_headlines)
    print(f"Weighted Aggregated Sentiment: {result}")
