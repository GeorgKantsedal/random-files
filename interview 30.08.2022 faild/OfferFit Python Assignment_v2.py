"""
Instructions:

Fill in the methods of the DataCleaner class to produce the same printed results
as in the comments below. Good luck, and have fun!
"""
from __future__ import annotations
import pdb

import numpy as np
import pandas as pd
import pickle

from typing import Dict, Any, List, Tuple


class DataCleaner:
    """
    Transform a pandas df while keeping track of the history of transformations to
    allow reverting back to earlier state.
    """
    def __init__(self, df: pd.DataFrame):
        self.current = df
        self.actions = []

    def adjust_dtype(self, types: Dict[str, Any]) -> DataCleaner:
        """Adjust customer types function."""
        self.current = self.current.astype(types)
        self.actions.append([self.revert_adjust_dtype,
            dict(self.current[types.keys()].dtypes),
            f'Adjusted dtypes using {types}'])
        return self

    @staticmethod
    def revert_adjust_dtype(df: pd.DataFrame, types: Dict[str, Any]) -> pd.DataFrame:
        """ Revert for adjust_dtype function."""
        return df.astype(types)

    def impute_missing(self, columns: List[str]) -> DataCleaner:
        """Impute missing function. It could be better?
        I feel yes, but I'm already tired of refactoring"""
        df = self.current
        nans = np.where(self.current.isna())
        df[columns] = df[columns].fillna(df[columns].mean())
        self.current = df

        self.actions.append([self.revert_impute_missing, nans, f'Imputed missing in {columns}'])
        return self

    @staticmethod
    def revert_impute_missing(df: pd.DataFrame, nans: Tuple(np.array, np.array)) -> pd.DataFrame:
        """ Revert for impute_missing function.
        I'm not shure about OOP Class style, maybe it should be in the end of it."""
        df.iloc[nans] = np.nan
        return df

    def revert(self, steps_back: int = 1) -> DataCleaner:
        """Revert works stap-by-step. I add retur in order to make chains of methods.
        As for me it more comfortable."""
        if len(self.actions) > steps_back and (steps_back > 0):
            for i in range(steps_back):
                self.current = self.actions[-1][0](self.current, self.actions[-1][1])
                self.actions = self.actions[:-1]
        else:
            print('Invalid step')
        return self

    def save(self, path: str) -> None:
        with open(path, 'wb') as f:        
            pickle.dump(self, f)

    @staticmethod
    def load(path: str) -> DataCleaner:
        """Picle loader, better then sqlite but could be a problem:
        if we will biuld child class or change version of python"""
        with open(path, 'rb') as f:
            obj = pickle.load(f)
        return obj

    @property
    def history(self) -> str:
        """Custom print in order to save memory"""
        history = []
        df = self.current
        for a in self.actions[::-1]:
            history.append(str(df)+'), ')
            history.append('('+a[2]+', ')
            df = a[0](df, a[1])
        history.append(str(df)+'), ')
        history.append('(Initial df, ')
        return '['+''.join(history[::-1])[:-2]+']'



transactions = pd.DataFrame(
    {
        "customer_id": [10, 10, 13, 10, 11, 11, 10],
        "amount": [1.00, 1.31, 20.5, 0.5, 0.2, 0.2, np.nan],
        "timestamp": [
            "2020-10-08 11:32:01",
            "2020-10-08 13:45:00",
            "2020-10-07 05:10:30",
            "2020-10-08 12:30:00",
            "2020-10-07 01:29:33",
            "2020-10-08 13:45:00",
            "2020-10-09 02:05:21",
        ]
    }
)
transactions_dc = DataCleaner(transactions)

print(f"Current dataframe:\n{transactions_dc.current}")

# Current dataframe:
#    customer_id  amount            timestamp
# 0           10    1.00  2020-10-08 11:32:01
# 1           10    1.31  2020-10-08 13:45:00
# 2           13   20.50  2020-10-07 05:10:30
# 3           10    0.50  2020-10-08 12:30:00
# 4           11    0.20  2020-10-07 01:29:33
# 5           11    0.20  2020-10-08 13:45:00
# 6           10     NaN  2020-10-09 02:05:21

print(f"Current dtypes:\n{transactions_dc.current.dtypes}")

# Initial dtypes:
# customer_id      int64
# amount         float64
# timestamp       object
# dtype: object

transactions_dc.adjust_dtype({"timestamp": np.datetime64})

print(f"Changed dtypes to:\n{transactions_dc.current.dtypes}")

# Changed dtypes to:
# customer_id             int64
# amount                float64
# timestamp      datetime64[ns]

transactions_dc.impute_missing(columns=["amount"])

print(f"Imputed missing as overall mean:\n{transactions_dc.current}")

# Imputed missing as mean:
#    customer_id     amount           timestamp
# 0           10   1.000000 2020-10-08 11:32:01
# 1           10   1.310000 2020-10-08 13:45:00
# 2           13  20.500000 2020-10-07 05:10:30
# 3           10   0.500000 2020-10-08 12:30:00
# 4           11   0.200000 2020-10-07 01:29:33
# 5           11   0.200000 2020-10-08 13:45:00
# 6           10   3.951667 2020-10-09 02:05:21

print(f"History of changes:\n{transactions_dc.history}")

# ** Any coherent structure with history of changes **
# E.g., here's one possibility

# History of changes:
# [('Initial df',    customer_id  amount            timestamp
# 0           10    1.00  2020-10-08 11:32:01
# 1           10    1.31  2020-10-08 13:45:00
# 2           13   20.50  2020-10-07 05:10:30
# 3           10    0.50  2020-10-08 12:30:00
# 4           11    0.20  2020-10-07 01:29:33
# 5           11    0.20  2020-10-08 13:45:00
# 6           10     NaN  2020-10-09 02:05:21), ("Adjusted dtypes using {'timestamp': <class 'numpy.datetime64'>}",    customer_id  amount           timestamp
# 0           10    1.00 2020-10-08 11:32:01
# 1           10    1.31 2020-10-08 13:45:00
# 2           13   20.50 2020-10-07 05:10:30
# 3           10    0.50 2020-10-08 12:30:00
# 4           11    0.20 2020-10-07 01:29:33
# 5           11    0.20 2020-10-08 13:45:00
# 6           10     NaN 2020-10-09 02:05:21), ("Imputed missing in ['amount']",    customer_id     amount           timestamp
# 0           10   1.000000 2020-10-08 11:32:01
# 1           10   1.310000 2020-10-08 13:45:00
# 2           13  20.500000 2020-10-07 05:10:30
# 3           10   0.500000 2020-10-08 12:30:00
# 4           11   0.200000 2020-10-07 01:29:33
# 5           11   0.200000 2020-10-08 13:45:00
# 6           10   3.951667 2020-10-09 02:05:21)]

transactions_dc.save("transactions")
loaded_dc = DataCleaner.load("transactions")
print(f"Loaded DataCleaner current df:\n{loaded_dc.current}")

# Loaded DataCleaner current df:
#    customer_id     amount           timestamp
# 0           10   1.000000 2020-10-08 11:32:01
# 1           10   1.310000 2020-10-08 13:45:00
# 2           13  20.500000 2020-10-07 05:10:30
# 3           10   0.500000 2020-10-08 12:30:00
# 4           11   0.200000 2020-10-07 01:29:33
# 5           11   0.200000 2020-10-08 13:45:00
# 6           10   3.951667 2020-10-09 02:05:21

transactions_dc.revert()
print(f"Reverting missing value imputation:\n{transactions_dc.current}")

# Reverting missing value imputation:
#    customer_id  amount           timestamp
# 0           10    1.00 2020-10-08 11:32:01
# 1           10    1.31 2020-10-08 13:45:00
# 2           13   20.50 2020-10-07 05:10:30
# 3           10    0.50 2020-10-08 12:30:00
# 4           11    0.20 2020-10-07 01:29:33
# 5           11    0.20 2020-10-08 13:45:00
# 6           10     NaN 2020-10-09 02:05:21
