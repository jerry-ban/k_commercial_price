__author__ = 'jerry.ban'

import time
import re

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


def find_in_str_ss2(row,two_words_re, ss2):
    for doc_word in two_words_re.finditer(row):
        print(doc_word)
        suggestion = ss2.best_word(doc_word.group(1), silent=True)
        if suggestion is not None:
            return doc_word.group(1)
    return ''

def find_in_list_ss1(list, ss1):
    for doc_word in list:
        suggestion = ss1.best_word(doc_word, silent=True)
        if suggestion is not None:
            return doc_word
    return ''

def find_in_list_ss2(list, ss2):
    for doc_word in list:
        suggestion = ss2.best_word(doc_word, silent=True)
        if suggestion is not None:
            return doc_word
    return ''


def process_with_regex(dataset ):
    start_time= time.time()

    #very expensive: , like: 14kt gold engagement ring
    karats_regex = r'(\d)([\s-]?)(karat|karats|carat|carats|kt)([^\w])'
    karats_repl = r'\1k\4'

    # glue unit with determiner glued together
    unit_regex = r'(\d+)[\s-]([a-z]{2})(\s)'
    unit_repl = r'\1\2\3'

    # dataset.loc[dataset['name'].str.contains(karats_regex), ["name"]]
    dataset['name'] = dataset['name'].str.replace(karats_regex, karats_repl)
    dataset['item_description'] = dataset['item_description'].str.replace(karats_regex, karats_repl)
    print("{} Karats normalized.".format(time.time()-start_time))

    dataset['name'] = dataset['name'].str.replace(unit_regex, unit_repl)
    dataset['item_description'] = dataset['item_description'].str.replace(unit_regex, unit_repl)
    #print(f'[{time() - start_time}] Units glued.')
    print('{} Units glued.'.format(time.time() - start_time))

    return dataset