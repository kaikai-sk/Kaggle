import numpy as np
import pandas as pd
import datetime
import gc
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')
np.random.seed(4590)

if __name__ == "__main__":
    df_train = pd.read_csv('../input/train.csv')
    print(df_train.head(5))
    df_test = pd.read_csv('../input/test.csv')
    print(df_test.head(5))
    df_hist_trans = pd.read_csv('../input/historical_transactions.csv')
    print(df_hist_trans.head(5))
    df_new_merchant_trans = pd.read_csv('../input/new_merchant_transactions.csv')
    print(df_new_merchant_trans.head(5))

    
