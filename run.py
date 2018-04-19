from func01 import read_train_file

import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import plotnine_examples as pe

from mpl_toolkits.basemap import Basemap
from matplotlib import cm

from sklearn.preprocessing import MinMaxScaler

file_train = "train.tsv"
# rows: 669959,70, 3665
train_raw_data = read_train_file(file_train)
train_raw_data.shape