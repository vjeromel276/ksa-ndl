#!/usr/bin/env python3
import pandas as pd
import numpy as np
from typing import Iterator, Tuple

class TimeSeriesSplitter:
    """
    A rolling/expanding-window time-series cross-validator.

    Parameters
    ----------
    train_window : int
        Number of consecutive time-steps (e.g. days) to use for training.
    test_window : int
        Number of consecutive time-steps to use for testing.
    step : int, default=1
        How many time-steps to advance the window each fold.
    """
    def __init__(self, train_window: int, test_window: int, step: int = 1):
        self.train_window = train_window
        self.test_window  = test_window
        self.step         = step

    def split(
        self,
        df: pd.DataFrame,
        date_col: str = "date"
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Yield train/test index arrays for each fold.

        Parameters
        ----------
        df : DataFrame
          Must contain a column `date_col` of Timestamps.
        date_col : str
          Name of the date column to sort by.

        Yields
        ------
        train_idx, test_idx : (np.ndarray, np.ndarray)
          Integer positional indices into `df`.
        """
        # 1) sort by date
        df2 = df.copy()
        df2[date_col] = pd.to_datetime(df2[date_col])
        df2 = df2.sort_values(date_col).reset_index(drop=True)

        # 2) unique ordered dates
        unique_dates = df2[date_col].unique()
        n_dates      = len(unique_dates)

        # 3) rolling windows
        start = 0
        while True:
            train_start = start
            train_end   = start + self.train_window - 1
            test_start  = train_end + 1
            test_end    = test_start + self.test_window - 1

            # stop if we overflow
            if test_end >= n_dates:
                break

            # which rows fall into these date-ranges?
            train_dates = unique_dates[train_start : train_end + 1]
            test_dates  = unique_dates[test_start  : test_end  + 1]

            train_idx = df2.index[df2[date_col].isin(train_dates)].to_numpy()
            test_idx  = df2.index[df2[date_col].isin(test_dates )].to_numpy()

            yield train_idx, test_idx
            start += self.step
