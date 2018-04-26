import preprocess

import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import plotnine_examples as pe

from mpl_toolkits.basemap import Basemap
from matplotlib import cm

from sklearn.preprocessing import MinMaxScaler



from training import *

file_train = "train.tsv"
# rows: 669959,70, 3665
rows= 10**4
df_testid, merge, y_train, nrow_train = preprocess.read_and_clean_data(rows)
print("data ({})processed, to train data next".format(merge.shape))
model_train(merge, y_train, nrow_train, df_testid)




