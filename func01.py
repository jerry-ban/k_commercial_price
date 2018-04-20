__author__ = 'jerry.ban'

import pandas as pd
import numpy as np
import gc
import os
import logging
import charset
import sys

file_train = "train.tsv"
file_test = "test.tsv"
def read_and_clean_data(data_count = None):
    if data_count is None:
        train = pd.read_table(file_train, engine='c', dtype=
                                {'item_condition_id': 'category',
                                 'shipping': 'category'}
                          )
    else:
        train = pd.read_table(file_train, engine='c', nrows=data_count , dtype=
                                {'item_condition_id': 'category',
                                 'shipping': 'category'}
                          )
    train = train[train.price>0]

    #based on below to decide how to fill or clean data
    train.shape
    train.isnull().sum()  # here we know the brand_name and category_name has missing values, need to handle

    y_train = np.log1p(train.price)

    if data_count is None:
        test= pd.read_table(file_test, engine='c', encoding = "windows-1252", dtype={'item_condition_id': 'category', 'shipping': 'category'} )

    else:
        test= pd.read_table(file_test, engine='c', nrows=data_count, dtype= {'item_condition_id': 'category', 'shipping': 'category'} )

    merged = pd.concat([train, test])

    del train
    del test
    gc.collect()

    merged['has_category'] = (merged['category_name'].notnull()).astype('category')
    merged["category_name"] = merged["category_name"].fillna("other/other/other").str.lower().astype("str")
    merged['cat_0'], merged['cat_1'], merged['cat_2'], merged['cat_0_1'] = \
        zip(*merged['category_name'].apply(lambda x: split_cat_to_new_ones(x)))

    merged["has_brand"] = (merged["brand_name"].notnull()).astype("category")
    merged['brand_name'] = merged['name'].fillna('').str.lower().astype("str")

    merged["cat_0_cond"] = merged["cat_0"].astype("str") + "_" + merged["item_condition_id"].astype("str")
    merged["cat_1_cond"] = merged["cat_1"].astype("str") + "_" + merged["item_condition_id"].astype("str")
    merged["cat_2_cond"] = merged["cat_2"].astype("str") + "_" + merged["item_condition_id"].astype("str")

    merged['name'] = merged['name'].fillna('').str.lower().astype("str")

    merged['item_description'] = merged['item_description'].fillna('').str.lower().replace("No description yet", "")
    #to view the text in windows with pycharm(format is windows-1252 or cp1252, need conver the text encoding to utf-8
    merged['item_description'].str.encode("utf-8")




    df=train
    return df


def split_cat_to_new_ones(text):
    default=['other']*3
    try:
        cats = text.split("/")
        if len(cats)<3:
            cats.extend(default)
            cats = cats[0:2]
        return cats[0], cats[1], cats[2], cats[0] + '/' + cats[1]
    except:
        print("no category")
        return default[0], default[1], default[2], default[0] + '/' + default[1]