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

from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn import ensemble, metrics

warnings.filterwarnings('ignore')
np.random.seed(4590)

def get_new_columns(name,aggs):
    return [name + '_' + k + '_' + agg for k in aggs.keys() for agg in aggs[k]]

if __name__ == "__main__":
    df_train = pd.read_csv('../input/train.csv')
    print(df_train.head(5))
    df_test = pd.read_csv('../input/test.csv')
    print(df_test.head(5))
    df_hist_trans = pd.read_csv('../input/historical_transactions.csv')
    print(df_hist_trans.head(5))
    df_new_merchant_trans = pd.read_csv('../input/new_merchant_transactions.csv')
    print(df_new_merchant_trans.head(5))

    for df in [df_hist_trans,df_new_merchant_trans]:
        df['category_2'].fillna(1.0,inplace=True)
        df['category_3'].fillna('A',inplace=True)
        df['merchant_id'].fillna('M_ID_00a6ca8a8a',inplace=True)

    for df in [df_hist_trans,df_new_merchant_trans]:
        df['purchase_date'] = pd.to_datetime(df['purchase_date'])
        df['year'] = df['purchase_date'].dt.year
        df['weekofyear'] = df['purchase_date'].dt.weekofyear
        df['month'] = df['purchase_date'].dt.month
        df['dayofweek'] = df['purchase_date'].dt.dayofweek
        df['weekend'] = (df.purchase_date.dt.weekday >=5).astype(int)
        df['hour'] = df['purchase_date'].dt.hour
        df['authorized_flag'] = df['authorized_flag'].map({'Y':1, 'N':0})
        df['category_1'] = df['category_1'].map({'Y':1, 'N':0})
        #https://www.kaggle.com/c/elo-merchant-category-recommendation/discussion/73244
        df['month_diff'] = ((datetime.datetime.today() - df['purchase_date']).dt.days)//30
        df['month_diff'] += df['month_lag']

    aggs = {}
    for col in ['month','hour','weekofyear','dayofweek','year','subsector_id','merchant_id','merchant_category_id']:
        aggs[col] = ['nunique']

    aggs['purchase_amount'] = ['sum','max','min','mean','var']
    aggs['installments'] = ['sum','max','min','mean','var']
    aggs['purchase_date'] = ['max','min']
    aggs['month_lag'] = ['max','min','mean','var']
    aggs['month_diff'] = ['mean']
    aggs['authorized_flag'] = ['sum', 'mean']
    aggs['weekend'] = ['sum', 'mean']
    aggs['category_1'] = ['sum', 'mean']
    aggs['card_id'] = ['size']

    for col in ['category_2','category_3']:
        df_hist_trans[col+'_mean'] = df_hist_trans.groupby([col])['purchase_amount'].transform('mean')
        aggs[col+'_mean'] = ['mean']

    new_columns = get_new_columns('hist',aggs)
    df_hist_trans_group = df_hist_trans.groupby('card_id').agg(aggs)
    df_hist_trans_group.columns = new_columns
    df_hist_trans_group.reset_index(drop=False,inplace=True)
    df_hist_trans_group['hist_purchase_date_diff'] = (df_hist_trans_group['hist_purchase_date_max'] - df_hist_trans_group['hist_purchase_date_min']).dt.days
    df_hist_trans_group['hist_purchase_date_average'] = df_hist_trans_group['hist_purchase_date_diff']/df_hist_trans_group['hist_card_id_size']
    df_hist_trans_group['hist_purchase_date_uptonow'] = (datetime.datetime.today() - df_hist_trans_group['hist_purchase_date_max']).dt.days
    df_train = df_train.merge(df_hist_trans_group,on='card_id',how='left')
    df_test = df_test.merge(df_hist_trans_group,on='card_id',how='left')

    del df_hist_trans_group;
    gc.collect()

    aggs = {}
    for col in ['month','hour','weekofyear','dayofweek','year','subsector_id','merchant_id','merchant_category_id']:
        aggs[col] = ['nunique']
    aggs['purchase_amount'] = ['sum','max','min','mean','var']
    aggs['installments'] = ['sum','max','min','mean','var']
    aggs['purchase_date'] = ['max','min']
    aggs['month_lag'] = ['max','min','mean','var']
    aggs['month_diff'] = ['mean']
    aggs['weekend'] = ['sum', 'mean']
    aggs['category_1'] = ['sum', 'mean']
    aggs['card_id'] = ['size']

    for col in ['category_2','category_3']:
        df_new_merchant_trans[col+'_mean'] = df_new_merchant_trans.groupby([col])['purchase_amount'].transform('mean')
        aggs[col+'_mean'] = ['mean']

    new_columns = get_new_columns('new_hist',aggs)
    df_hist_trans_group = df_new_merchant_trans.groupby('card_id').agg(aggs)
    df_hist_trans_group.columns = new_columns
    df_hist_trans_group.reset_index(drop=False,inplace=True)
    df_hist_trans_group['new_hist_purchase_date_diff'] = (df_hist_trans_group['new_hist_purchase_date_max'] - df_hist_trans_group['new_hist_purchase_date_min']).dt.days
    df_hist_trans_group['new_hist_purchase_date_average'] = df_hist_trans_group['new_hist_purchase_date_diff']/df_hist_trans_group['new_hist_card_id_size']
    df_hist_trans_group['new_hist_purchase_date_uptonow'] = (datetime.datetime.today() - df_hist_trans_group['new_hist_purchase_date_max']).dt.days
    df_train = df_train.merge(df_hist_trans_group,on='card_id',how='left')
    df_test = df_test.merge(df_hist_trans_group,on='card_id',how='left')
    del df_hist_trans_group;gc.collect()

    del df_hist_trans;gc.collect()
    del df_new_merchant_trans;gc.collect()
    df_train.head(5)

    train_y = df_train['target']
    train_x = df_train.drop('target', axis=1)

    rf = ensemble.RandomForestRegressor(#bootstrap=best_parms['bootstrap'],
                                    max_depth=4,#best_parms['max_depth'],
                                    max_features='auto',#best_parms['max_features'],
                                    min_samples_leaf=30,#best_parms['min_samples_leaf'],
                                    n_estimators=2500)#best_parms['n_estimators'])

    rf.fit(train_x,train_y)

    test_y = df_test['failure']
    test_x = df_test.drop('failure', axis=1)

    test_class_preds = rf.predict(test_x)
    train_class_preds = rf.predict(train_x)

    train_rmse = np.sqrt(metrics.mean_squared_error(train_y,train_class_preds))
    test_rmse = np.sqrt(metrics.mean_squared_error(test_y,test_class_preds))

    print("train_rmse : ",train_rmse , "test_rmse : " , test_rmse)
