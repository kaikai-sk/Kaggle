import numpy as np
import pandas as pd
import datetime
import gc
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
import xgboost as xgb
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
    # print(df_train.head(5))
    df_test = pd.read_csv('../input/test.csv')
    # print(df_test.head(5))
    df_hist_trans = pd.read_csv('../input/historical_transactions.csv')
    # print(df_hist_trans.head(5))
    df_new_merchant_trans = pd.read_csv('../input/new_merchant_transactions.csv')
    # print(df_new_merchant_trans.head(5))

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

    del df_hist_trans;
    gc.collect()
    del df_new_merchant_trans;
    gc.collect()

    # feature_is_time = ['first_active_month','hist_purchase_date_max','hist_purchase_date_min','new_hist_purchase_date_max',
    #     'new_hist_purchase_date_min']
    #
    # df_train.drop(columns = feature_is_time,inplace=True)
    # df_test.drop(columns = feature_is_time,inplace=True)

    print(df_train.head(5))
    df_train.to_csv('df_train_head.csv',index=False)

    df_train['outliers'] = 0
    df_train.loc[df_train['target'] < -30, 'outliers'] = 1
    df_train['outliers'].value_counts()

    for df in [df_train,df_test]:
        df['first_active_month'] = pd.to_datetime(df['first_active_month'])
        df['dayofweek'] = df['first_active_month'].dt.dayofweek
        df['weekofyear'] = df['first_active_month'].dt.weekofyear
        df['month'] = df['first_active_month'].dt.month
        df['elapsed_time'] = (datetime.datetime.today() - df['first_active_month']).dt.days
        df['hist_first_buy'] = (df['hist_purchase_date_min'] - df['first_active_month']).dt.days
        df['new_hist_first_buy'] = (df['new_hist_purchase_date_min'] - df['first_active_month']).dt.days
        for f in ['hist_purchase_date_max','hist_purchase_date_min','new_hist_purchase_date_max',\
                         'new_hist_purchase_date_min']:
            df[f] = df[f].astype(np.int64) * 1e-9
        df['card_id_total'] = df['new_hist_card_id_size']+df['hist_card_id_size']
        df['purchase_amount_total'] = df['new_hist_purchase_amount_sum']+df['hist_purchase_amount_sum']

    for f in ['feature_1','feature_2','feature_3']:
        order_label = df_train.groupby([f])['outliers'].mean()
        df_train[f] = df_train[f].map(order_label)
        df_test[f] = df_test[f].map(order_label)

    df_train_columns = [c for c in df_train.columns if c not in ['card_id', 'first_active_month','target','outliers']]

    target = df_train['target']
    del df_train['target']

    """
        这里是lightgbm model
    """
    param = {'num_leaves': 31,
         'min_data_in_leaf': 30,
         'objective':'regression',
         'max_depth': -1,
         'learning_rate': 0.01,
         "min_child_samples": 20,
         "boosting": "gbdt",
         "feature_fraction": 0.9,
         "bagging_freq": 1,
         "bagging_fraction": 0.9 ,
         "bagging_seed": 11,
         "metric": 'rmse',
         "lambda_l1": 0.1,
         "verbosity": -1,
         "nthread": 4,
         "random_state": 4590}
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=4590)
    oof = np.zeros(len(df_train))
    lgbm_predictions = np.zeros(len(df_test))
    feature_importance_df = pd.DataFrame()

    for fold_, (trn_idx, val_idx) in enumerate(folds.split(df_train,df_train['outliers'].values)):
        print("fold {}".format(fold_))
        trn_data = lgb.Dataset(df_train.iloc[trn_idx][df_train_columns], label=target.iloc[trn_idx])#, categorical_feature=categorical_feats)
        val_data = lgb.Dataset(df_train.iloc[val_idx][df_train_columns], label=target.iloc[val_idx])#, categorical_feature=categorical_feats)

        num_round = 10000
        clf = lgb.train(param, trn_data, num_round, valid_sets = [trn_data, val_data], verbose_eval=100, early_stopping_rounds = 100)
        oof[val_idx] = clf.predict(df_train.iloc[val_idx][df_train_columns], num_iteration=clf.best_iteration)

        fold_importance_df = pd.DataFrame()
        fold_importance_df["Feature"] = df_train_columns
        fold_importance_df["importance"] = clf.feature_importance()
        fold_importance_df["fold"] = fold_ + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

        lgbm_predictions += clf.predict(df_test[df_train_columns], num_iteration=clf.best_iteration) / folds.n_splits

    np.sqrt(mean_squared_error(oof, target))

    cols = (feature_importance_df[["Feature", "importance"]]
        .groupby("Feature")
        .mean()
        .sort_values(by="importance", ascending=False)[:1000].index)

    best_features = feature_importance_df.loc[feature_importance_df.Feature.isin(cols)]

    plt.figure(figsize=(14,25))
    sns.barplot(x="importance",
                y="Feature",
                data=best_features.sort_values(by="importance",
                                               ascending=False))
    plt.title('LightGBM Features (avg over folds)')
    plt.tight_layout()
    plt.savefig('lgbm_importances.png')

    """
        xgb 模型
    """
    xgb_params = {'eta': 0.001, 'max_depth': 7, 'subsample': 0.8, 'colsample_bytree': 0.8,
          'objective': 'reg:linear', 'eval_metric': 'rmse', 'silent': True}

    FOLDs = StratifiedKFold(n_splits=5, shuffle=True, random_state=4590)
    #KFold(n_splits=5, shuffle=True, random_state=1989)

    oof_xgb = np.zeros(len(df_train))
    xgb_predictions = np.zeros(len(df_test))


    for fold_, (trn_idx, val_idx) in enumerate(FOLDs.split(df_train,df_train['outliers'].values))):
        trn_data = xgb.DMatrix(data=train.iloc[trn_idx], label=target.iloc[trn_idx])
        val_data = xgb.DMatrix(data=train.iloc[val_idx], label=target.iloc[val_idx])
        watchlist = [(trn_data, 'train'), (val_data, 'valid')]
        print("xgb " + str(fold_) + "-" * 50)
        num_round = 2000
        xgb_model = xgb.train(xgb_params, trn_data, num_round, watchlist, early_stopping_rounds=100, verbose_eval=200)
        oof_xgb[val_idx] = xgb_model.predict(xgb.DMatrix(train.iloc[val_idx]), ntree_limit=xgb_model.best_ntree_limit+50)

        xgb_predictions += xgb_model.predict(xgb.DMatrix(test), ntree_limit=xgb_model.best_ntree_limit+50) / FOLDs.n_splits

    np.sqrt(mean_squared_error(oof_xgb, target))

    """
        随机森林模型
    """
    train_y = target
    train_x = df_train.values

    rf = ensemble.RandomForestRegressor(#bootstrap=best_parms['bootstrap'],
                                    max_depth=4,#best_parms['max_depth'],
                                    max_features='auto',#best_parms['max_features'],
                                    min_samples_leaf=30,#best_parms['min_samples_leaf'],
                                    n_estimators=2500)#best_parms['n_estimators'])

    rf.fit(train_x,train_y)

    # test_y = df_test['failure']
    test_x = df_test[df_train_columns]

    rf_predictions = rf.predict(test_x)



    # train_rmse = np.sqrt(metrics.mean_squared_error(train_y,train_class_preds))
    # test_rmse = np.sqrt(metrics.mean_squared_error(test_y,test_class_preds))
    #
    # print("train_rmse : ",train_rmse , "test_rmse : " , test_rmse)


    predictions = (lgbm_predictions + xgb_predictions) / 2

    sub_df = pd.DataFrame({"card_id":df_test["card_id"].values})
    sub_df["target"] = predictions
    sub_df.to_csv("submission.csv", index=False)
