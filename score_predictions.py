#!/usr/bin/env python3

import argparse
import logging
import pandas as pd

# Configure debug logging for every transformation step

def configure_logging():
    logger = logging.getLogger(__name__)
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s %(levelname)s:%(name)s:%(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    return logger

logger = configure_logging()


def main():
    parser = argparse.ArgumentParser(
        description="Score predictions and save to CSV"
    )
    parser.add_argument(
        "--input", "-i", required=True,
        help="Path to input predictions CSV file"
    )
    parser.add_argument(
        "--output", "-o", required=True,
        help="Path for output scored CSV file"
    )
    args = parser.parse_args()

    # Load data
    logger.debug("Loading predictions from %s", args.input)
    df = pd.read_csv(args.input)
    logger.debug("Loaded %d rows; columns: %s", len(df), list(df.columns))

    # Drop the 'signal' column if present
    if 'signal' in df.columns:
        logger.debug("Dropping 'signal' column")
        df = df.drop(columns=['signal'])
        logger.debug("Columns now: %s", list(df.columns))

    # Compute product score
    logger.debug("Computing product score: p_up * pred_return")
    df['score_product'] = df['p_up'] * df['pred_return']
    logger.debug("Added 'score_product' column")

    # Compute signed confidence score
    logger.debug("Computing signed confidence score: (2*p_up - 1) * pred_return")
    df['score_signed_confidence'] = (2 * df['p_up'] - 1) * df['pred_return']
    logger.debug("Added 'score_signed_confidence' column")

    # Sort descending by signed confidence
    logger.debug("Sorting by 'score_signed_confidence' descending")
    df = df.sort_values(by='score_signed_confidence', ascending=False)
    logger.debug("Top 5 rows after sort:\n%s", df.head())

    # Save output
    logger.debug("Saving scored predictions to %s", args.output)
    df.to_csv(args.output, index=False)
    logger.debug("Finished saving %d rows", len(df))


if __name__ == '__main__':
    main()
