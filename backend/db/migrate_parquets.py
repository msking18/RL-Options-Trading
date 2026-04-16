import pandas as pd
import os
import logging
from backend.db.local_db_manager import LocalDBManager

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("DataMigration")

def migrate():
    db = LocalDBManager()
    
    # 1. Migrate Macro Data
    macro_path = "data/historical/macro_data.parquet"
    if os.path.exists(macro_path):
        try:
            logger.info("Reading macro_data.parquet...")
            macro_df = pd.read_parquet(macro_path)
            # Ensure Date is in correct format
            if 'Date_Only' in macro_df.columns and 'Date' not in macro_df.columns:
                macro_df = macro_df.rename(columns={'Date_Only': 'date'})
            elif 'Date' in macro_df.columns:
                macro_df = macro_df.rename(columns={'Date': 'date'})
            
            # Remove any columns not in schema if necessary, but save_macro_data handles case sensitivity
            db.save_macro_data(macro_df)
            logger.info("Successfully migrated macro data.")
        except Exception as e:
            logger.error(f"Error migrating macro data: {e}")
    else:
        logger.warning(f"Macro data file not found at {macro_path}")

    # 2. Migrate Economic Events
    events_path = "data/historical/economic_events.parquet"
    if os.path.exists(events_path):
        try:
            logger.info("Reading economic_events.parquet...")
            events_df = pd.read_parquet(events_path)
            db.save_economic_events(events_df)
            logger.info("Successfully migrated economic events.")
        except Exception as e:
            logger.error(f"Error migrating economic events: {e}")
    else:
        logger.warning(f"Economic events file not found at {events_path}")

    final_stats = db.get_headline_stats()
    logger.info(f"Final DB Stats: {final_stats}")
    db.close()

if __name__ == "__main__":
    migrate()
