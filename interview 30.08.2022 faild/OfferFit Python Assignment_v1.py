"""
Instructions:

Fill in the methods of the DataCleaner class to produce the same printed results
as in the comments below. Good luck, and have fun!
"""
from __future__ import annotations
from pdb import Pdb
import pdb

import numpy as np
import pandas as pd
import sqlite3
import shutil
import time 
import os

from typing import Dict, Any, List


class DataCleaner:
    """
    Transform a pandas df while keeping track of the history of transformations to
    allow reverting back to earlier state.
    """
    def __init__(self, df: pd.DataFrame, history_path: str = None, actions: List[str] = []):
        # ** Your code here **
        self.current = df
        self.history_path = f'tmp{time.time()}' if history_path is None else history_path
        self.actions = actions

        self.log_change('Initial df')

    def log_change(self, change: str):
        """Logs changes to ROM.

        Args:
            change (str): change discription
        """
        self.actions.append(change)
        conn = sqlite3.connect(self.history_path)
        self.current.to_sql(f'act_{len(self.actions)}', con=conn, if_exists='replace')
        conn.close()

    def adjust_dtype(self, types: Dict[str, Any]) -> None:
        """Changes type for selected columns.

        Args:
            types (Dict[str, Any]): Dict of column names and new types
        """
        self.current = self.current.astype(types)

        self.log_change(f'Adjusted dtypes using {types}')
    

    def impute_missing(self, columns: List[str]) -> None:
        """Fills missing as overall mean for selected columns

        Args:
            columns (List[str]): List of column names
        """
        df = self.current
        df[columns] = df[columns].fillna(df[columns].mean())
        self.current = df

        self.log_change(f'Imputed missing in {columns}')

    def revert(self, steps_back: int = 1) -> None:
        """Revert DataFrame steps_back. Reverted changes are unavailable

        Args:
            steps_back (int, optional): Number of steps to revert. Defaults to 1.
        """
        if len(self.history) > steps_back and (steps_back > 0):
            self.actions = self.actions[:-steps_back]
            conn = sqlite3.connect(self.history_path)
            self.current = pd.read_sql_query(f'select * from act_{len(self.actions)}', con=conn)
            conn.cursor().execute(f'DROP TABLE act_{len(self.actions)+1}')
            conn.close()
        else:
            print('Invalid step')

    def save(self, path: str) -> None:
        """Save obj to the file

        Args:
            path (str): Path with file name on the end
        """

        shutil.copy2(self.history_path, path)
        conn = sqlite3.connect(path)
        pd.DataFrame({'action': self.actions}).to_sql(f'actions', con=conn, if_exists='replace')
        conn.close()

    @property
    def history(self) -> str:
        """In order to minimize RAM we need this function to print history.

        Returns:
            str: All changes and data states
        """
        history = '['
        conn = sqlite3.connect(self.history_path)
        for a in range(1, len(self.actions)+1):
            history += f'({self.actions[a-1]},'
            history += str(pd.read_sql_query(f'select * from act_{a}', con=conn)) 
            history += ')]' if a == len(self.actions) else '), '
        conn.close()
        return history

    @staticmethod
    def load(path: str) -> DataCleaner:
        """Obj loader.

        Args:
            path (str): Path with file name on the end

        Returns:
            DataCleaner: Loaded obj
        """

        conn = sqlite3.connect(path)
        actions = pd.read_sql_query('select * from actions', con=conn)['action'].to_list()
        
        curent = pd.read_sql_query(f'select * from act_{len(actions)}', con=conn)
        # import pdb; pdb.set_trace()
        history_path = f'tmp {time.time()}'
        conn.close()

        shutil.copy2(path, history_path)
        return DataCleaner(curent, history_path=history_path, actions=actions)
    
    def __del__(self):
        """Destructor.
        """
        os.remove(self.history_path)


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

print(str(transactions))

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

transactions_dc.revert(2) # It was (). I think this is better.
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
