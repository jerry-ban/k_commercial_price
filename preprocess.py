__author__ = 'jerry.ban'

import pandas as pd
import numpy as np
import gc
import re
import os
import logging
import charset
import sys
import time

import preprocess_func as routine
from cls_symspell import SymSpell
file_train = "train.tsv"
file_test = "test.tsv"

#CC_BRAND_WORD = r"[a-z0-9*/+\-'’?!.,|&%®™τθιό]+"
CC_BRAND_WORD  = r"[a-z0-9*/+\-'?!.,|&%]+"
#CC_TWO_WORDS_PATTERN = r"(?=(\s[a-z0-9*/+\-'’?!.,|&%®™τθιό]+\s[a-z0-9*/+\-'’?!.,|&%®™τθιό]+))"
CC_TWO_WORDS_PATTERN  = r"(?=(\s[a-z0-9*/+\-'?!.,|&%]+\s[a-z0-9*/+\-'?!.,|&%]+))"
#CC_BRAND_NAME_PATERN = r"^[a-z0-9*/+\-'’?!.,|&%®™τθιό]+\s[a-z0-9*/+\-'’?!.,|&%®™τθιό]+"
CC_BRAND_NAME_PATERN  = r"^[a-z0-9*/+\-'?!.,|&%]+\s[a-z0-9*/+\-'?!.,|&%]+"
#CC_BRAND_DESC_PATTERN = r"^[a-z0-9*/+\-'’?!.,|&%®™τθιό]+\s[a-z0-9*/+\-'’?!.,|&%®™τθιό]+"
CC_BRAND_DESC_PATTERN  = r"^[a-z0-9*/+\-'?!.,|&%]+\s[a-z0-9*/+\-'?!.,|&%]+"

def read_and_clean_data(data_count = None):
    start_time = time.time()
    data_count = None
    print("reading data...")
    if data_count is None:
        train = pd.read_table(file_train, engine='c', dtype={'item_condition_id': 'category','shipping': 'category'}
                              , index_col = None
                          )
    else:
        train = pd.read_table(file_train, engine='c', nrows=data_count , dtype={'item_condition_id': 'category', 'shipping': 'category'}
                              ,index_col = None
                        )
    #only keep valid price
    train = train[train.price>0]

    #based on below to decide how to fill or clean data

    train.shape
    train.isnull().sum()  # here we know the brand_name and category_name has missing values, need to handle

    nrow_train = train.shape[0]
    #transform price with log
    y_train = np.log1p(train.price)

    if data_count is None:
        test= pd.read_table(file_test, engine='c', dtype={'item_condition_id': 'category', 'shipping': 'category'} )

    else:
        test= pd.read_table(file_test, engine='c', nrows=data_count, dtype= {'item_condition_id': 'category', 'shipping': 'category'} )

    submission= test[['test_id']]

    merged = pd.concat([train, test], ignore_index=True)

    del train
    del test
    gc.collect()

    print("... {:<10.1f} process data...".format(time.time() - start_time))
    # create has_category feature, and split whole combined category into 3 specific categories
    # fill missing info for category_name
    merged['has_category'] = (merged['category_name'].notnull()).astype('category')
    merged["category_name"] = merged["category_name"].fillna("other/other/other").str.lower().astype("str")
    merged['cat_0'], merged['cat_1'], merged['cat_2'], merged['cat_0_1'] = \
        zip(*merged['category_name'].apply(lambda x: routine.split_cat_to_new_ones(x)))

    # create has_brand feature
    # fill missing info for brand_name
    merged["has_brand"] = (merged["brand_name"].notnull()).astype("category")
    merged['brand_name'] = merged['brand_name'].fillna('').str.lower().astype("str")

    # create new features: combine each category and condition
    merged["cat_0_cond"] = merged["cat_0"].astype("str") + "_" + merged["item_condition_id"].astype("str")
    merged["cat_1_cond"] = merged["cat_1"].astype("str") + "_" + merged["item_condition_id"].astype("str")
    merged["cat_2_cond"] = merged["cat_2"].astype("str") + "_" + merged["item_condition_id"].astype("str")

    #fill missing info for name
    merged['name'] = merged['name'].fillna('').str.lower().astype("str")

    #fill missing info for item_description
    merged['item_description'] = merged['item_description'].fillna('').str.lower().replace("No description yet", "")

    #to view the text in windows with pycharm(format is windows-1252 or cp1252, need conver the text encoding to utf-8
    merged['item_description'].str.encode("utf-8")

    #normalize expensive stuff, and units
    print("... {:<10.1f} re data...".format(time.time() - start_time))
    merged = routine.process_with_regex(merged)

    # fill in missing brand from name and description
    print("... {:<10.1f} filling missing brand data...".format(time.time() - start_time))
    merged = brands_filling(merged)

    # concancenat name and brand
    merged['name'] = merged['name'] + ' ' + merged['brand_name']
    merged['item_description'] = merged['item_description'] \
                                + ' ' + merged['name'] \
                                + ' ' + merged['cat_1'] \
                                + ' ' + merged['cat_2'] \
                                + ' ' + merged['cat_0'] \
                                + ' ' + merged['brand_name']
    ###print(f'[{time() - start_time}] Item description concatenated.')

    merged.drop(['price', 'test_id', 'train_id'], axis=1, inplace=True)
    print("... {:<10.1f} data processed".format(time.time() - start_time))
    return submission, merged, y_train, nrow_train


def brands_filling(df):
    vc = df['brand_name'].value_counts()
    brands = vc[vc > 0].index
    brand_word = CC_BRAND_WORD  # r"[a-z0-9*/+\-'’?!.,|&%®™τθιό]+"

    many_w_brands = brands[brands.str.contains(' ')]
    one_w_brands = brands[~brands.str.contains(' ')]

    ss2 = SymSpell(max_edit_distance=0)
    ss2.create_dictionary_from_arr(many_w_brands, token_pattern=r'.+')

    ss1 = SymSpell(max_edit_distance=0)
    ss1.create_dictionary_from_arr(one_w_brands, token_pattern=r'.+')

    two_words_re = re.compile(CC_TWO_WORDS_PATTERN) #r"(?=(\s[a-z0-9*/+\-'’?!.,|&%®™τθιό]+\s[a-z0-9*/+\-'’?!.,|&%®™τθιό]+))"

    print("Before empty brand_name: {}".format(len(df[df['brand_name'] == ''].index)))  # sum(df['brand_name'] == '')

    # r"^[a-z0-9*/+\-'’?!.,|&%®™τθιό]+\s[a-z0-9*/+\-'’?!.,|&%®™τθιό]+"
    n_name = df[df['brand_name'] == '']['name'].str.findall(pat=CC_BRAND_NAME_PATERN)

    df.loc[df['brand_name'] == '', 'brand_name'] = [routine.find_in_list_ss2(row, ss2) for row in n_name]

    #r"^[a-z0-9*/+\-'’?!.,|&%®™τθιό]+\s[a-z0-9*/+\-'’?!.,|&%®™τθιό]+"
    n_desc = df[df['brand_name'] == '']['item_description'].str.findall(pat=CC_BRAND_DESC_PATTERN)

    df.loc[df['brand_name'] == '', 'brand_name'] = [routine.find_in_list_ss2(row,ss2) for row in n_desc]

    n_name = df[df['brand_name'] == '']['name'].str.findall(pat=brand_word)
    df.loc[df['brand_name'] == '', 'brand_name'] = [routine.find_in_list_ss1(row,ss1) for row in n_name]

    desc_lower = df[df['brand_name'] == '']['item_description'].str.findall(pat=brand_word)
    df.loc[df['brand_name'] == '', 'brand_name'] = [routine.find_in_list_ss1(row,ss1) for row in desc_lower]

    print("After empty brand_name: {}".format(len(df[df['brand_name'] == ''].index)))

    del ss1, ss2
    gc.collect()

    return df
