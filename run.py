import func01

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
rows= 10**5
data = func01.read_and_clean_data(rows)
data.shape


