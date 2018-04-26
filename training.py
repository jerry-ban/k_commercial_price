__author__ = 'jerry.ban'

import sklearn.pipeline as ppline
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer, TfidfTransformer
import sklearn.preprocessing as skpreprocess
from sklearn.linear_model import Ridge
from scipy.sparse import csr_matrix
import numpy as np

import gc

import cls_itemselector as myclass
from cls_itemselector import ItemSelector, DropColumnsByDf
from preprocess_func import split_cat_to_new_ones
import time

def model_train(merge, y_train, nrow_train, submission):
    print("to train model...")
    start_time = time.time()
    meta_params = {'name_ngram': (1, 2),
                       'name_max_f': 75000,
                       'name_min_df': 10,

                       'category_ngram': (2, 3),
                       'category_token': '.+',
                       'category_min_df': 10,

                       'brand_min_df': 10,

                       'desc_ngram': (1, 3),
                       'desc_max_f': 150000,
                       'desc_max_df': 0.5,
                       'desc_min_df': 10}

    stopwords = frozenset(['the', 'a', 'an', 'is', 'it', 'this', ])

    vectorizer = ppline.FeatureUnion([
            ('name', ppline.Pipeline([
                ('select', myclass.ItemSelector('name')),
                ('transform', HashingVectorizer(
                    ngram_range=(1, 2),
                    n_features=2 ** 27,
                    norm='l2',
                    lowercase=False,
                    stop_words=stopwords
                )),
                ('drop_cols', myclass.DropColumnsByDf(min_df=2))
            ])),
            ('category_name', ppline.Pipeline([
                ('select', myclass.ItemSelector('category_name')),
                ('transform', HashingVectorizer(
                    ngram_range=(1, 1),
                    token_pattern='.+',
                    tokenizer= split_cat_to_new_ones,
                    n_features=2 ** 27,
                    norm='l2',
                    lowercase=False
                )),
                ('drop_cols',  myclass.DropColumnsByDf(min_df=2))
            ])),
            ('brand_name', ppline.Pipeline([
                ('select', ItemSelector('brand_name')),
                ('transform', CountVectorizer(
                    token_pattern='.+',
                    min_df=2,
                    lowercase=False
                )),
            ])),
            ('cat_0_cond', ppline.Pipeline([
                ('select',  myclass.ItemSelector('cat_0_cond')),
                ('transform', CountVectorizer(
                    token_pattern='.+',
                    min_df=2,
                    lowercase=False
                )),
            ])),
            ('cat_1_cond', ppline.Pipeline([
                ('select',  myclass.ItemSelector('cat_1_cond')),
                ('transform', CountVectorizer(
                    token_pattern='.+',
                    min_df=2,
                    lowercase=False
                )),
            ])),
            ('cat_2_cond', ppline.Pipeline([
                ('select',  myclass.ItemSelector('cat_2_cond')),
                ('transform', CountVectorizer(
                    token_pattern='.+',
                    min_df=2,
                    lowercase=False
                )),
            ])),
            ('has_brand', ppline.Pipeline([
                ('select',  myclass.ItemSelector('has_brand')),
                ('ohe', skpreprocess.OneHotEncoder())
            ])),
            ('shipping', ppline.Pipeline([
                ('select',  myclass.ItemSelector('shipping')),
                ('ohe', skpreprocess.OneHotEncoder())
            ])),
            ('item_condition_id', ppline.Pipeline([
                ('select',  myclass.ItemSelector('item_condition_id')),
                ('ohe', skpreprocess.OneHotEncoder())
            ])),
            ('item_description', ppline.Pipeline([
                ('select',  myclass.ItemSelector('item_description')),
                ('hash', HashingVectorizer(
                    ngram_range=(1, 3),
                    n_features=2 ** 27,
                    dtype=np.float32,
                    norm='l2',
                    lowercase=False,
                    stop_words=stopwords
                )),
                ('drop_cols',  myclass.DropColumnsByDf(min_df=2)),
            ]))
        ], n_jobs=1)

    print("... {:<10.1f} transform sparse...".format(time.time() - start_time))
    sparse_merge = vectorizer.fit_transform(merge)
        ###print(f'[{time() - start_time}] Merge vectorized')
    print(sparse_merge.shape)
    print("... {:<10.1f} transform tfidf...".format(time.time() - start_time))

    tfidf_transformer = TfidfTransformer()
    X = tfidf_transformer.fit_transform(sparse_merge)
    ###print(f'[{time() - start_time}] TF/IDF completed')

    X_train = X[:nrow_train]
    print(X_train.shape)

    X_test = X[nrow_train:]
    del merge
    del sparse_merge
    del vectorizer
    del tfidf_transformer
    gc.collect()

    print("... {:<10.1f} intersect drop columns...".format(time.time() - start_time))
    X_train, X_test = intersect_drop_columns(X_train, X_test, min_df=1)
    ###print(f'[{time() - start_time}] Drop only in train or test cols: {X_train.shape[1]}')
    gc.collect()

    print("... {:<10.1f} Ridge fit...".format(time.time() - start_time))
    ridge = Ridge(solver='auto', fit_intercept=True, alpha=0.4, max_iter=200, normalize=False, tol=0.01)
    ridge.fit(X_train, y_train)
    ###print(f'[{time() - start_time}] Train Ridge completed. Iterations: {ridge.n_iter_}')

    print("... {:<10.1f} Ridge predict...".format(time.time() - start_time))
    predsR = ridge.predict(X_test)
    ###print(f'[{time() - start_time}] Predict Ridge completed.')

    print("... {:<10.1f} output...".format(time.time() - start_time))
    submission.loc[:, 'price'] = np.expm1(predsR)
    submission.loc[submission['price'] < 0.0, 'price'] = 0.0
    submission.to_csv("submission_ridge.csv", index=False)
    print("... {:<10.1f} prediction work done".format(time.time() - start_time))

def intersect_drop_columns(train: csr_matrix, valid: csr_matrix, min_df=0):
    t = train.tocsc()
    v = valid.tocsc()
    nnz_train = ((t != 0).sum(axis=0) >= min_df).A1
    nnz_valid = ((v != 0).sum(axis=0) >= min_df).A1
    nnz_cols = nnz_train & nnz_valid
    res = t[:, nnz_cols], v[:, nnz_cols]
    return res