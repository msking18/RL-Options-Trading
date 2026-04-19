"""
One-time batch scorer for all unscored headlines in SQLite.
Uses the VADER-based SentimentEngine to score each headline.
"""
import sys
import os
import logging
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from backend.db.local_db_manager import LocalDBManager
from backend.sentiment.sentiment_engine import SentimentEngine

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("batch_scorer.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

BATCH_SIZE = 500


def run_batch_scoring():
    db = LocalDBManager()
    engine = SentimentEngine()

    # Get initial stats
    stats = db.get_headline_stats()
    unscored = stats["total_headlines"] - stats["scored_headlines"]
    logger.info(f"Starting batch scoring. {unscored} unscored headlines out of {stats['total_headlines']} total.")

    if unscored == 0:
        logger.info("All headlines are already scored. Nothing to do.")
        db.close()
        return

    total_scored = 0

    while True:
        # Fetch a batch of unscored headlines
        batch = db.get_unscored_headlines(batch_size=BATCH_SIZE)
        if batch.empty:
            break

        # Score each headline
        score_updates = []
        for _, row in batch.iterrows():
            scores = engine.analyze_headline(row["headline"])
            compound = scores["compound"]
            score_updates.append((compound, row["id"]))

        # Batch update DB
        db.update_sentiment_scores(score_updates)
        total_scored += len(score_updates)

        logger.info(f"Scored {total_scored} headlines so far...")

    # Final stats
    final_stats = db.get_headline_stats()
    final_unscored = final_stats["total_headlines"] - final_stats["scored_headlines"]
    logger.info(
        f"Batch scoring complete. "
        f"Total scored: {total_scored}. "
        f"Remaining unscored: {final_unscored}."
    )

    db.close()


if __name__ == "__main__":
    run_batch_scoring()
