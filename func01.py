__author__ = 'jerry.ban'

import pandas as pd
import os

def read_train_file(file_name, data_count = None):
    df = None
    train = pd.read_table(file_name, engine='c', dtype=
                                {'item_condition_id': 'category',
                                 'shipping': 'category'}
                          )

    df=train
    return df