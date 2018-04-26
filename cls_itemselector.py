__author__ = 'jerry.ban'

import pandas.api.types as spi_types
import sklearn.base as skbase

import time

class ItemSelector(skbase.BaseEstimator, skbase.TransformerMixin):
    def __init__(self, field):
        self.field = field
        self.start_time = 0

    def fit(self, x, y=None):
        return self

    def transform(self, dataframe):
        ###print(f'[{time()-self.start_time}] select {self.field}')
        dt = dataframe[self.field].dtype
        if spi_types.is_categorical_dtype(dt):
            return dataframe[self.field].cat.codes[:, None]
        elif spi_types.is_numeric_dtype(dt):
            return dataframe[self.field][:, None]
        else:
            return dataframe[self.field]



class DropColumnsByDf(skbase.BaseEstimator, skbase.TransformerMixin):
    def __init__(self, min_df=1, max_df=1.0):
        self.min_df = min_df
        self.max_df = max_df

    def fit(self, X, y=None):
        m = X.tocsc()
        self.nnz_cols = ((m != 0).sum(axis=0) >= self.min_df).A1
        if self.max_df < 1.0:
            max_df = m.shape[0] * self.max_df
            self.nnz_cols = self.nnz_cols & ((m != 0).sum(axis=0) <= max_df).A1
        return self

    def transform(self, X, y=None):
        m = X.tocsc()
        return m[:, self.nnz_cols]