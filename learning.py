import os
import random
import re
import warnings

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from keras.callbacks import EarlyStopping
from keras.utils import np_utils
from keras.models import load_model
from scipy.optimize import minimize
from sklearn.decomposition import NMF, PCA, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import f1_score, log_loss
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.random import set_seed
from tqdm.notebook import tqdm
import pickle

def load_preprocessed_data():
    f = open("./preprocessed_data/train_list.pickle","rb")
    x_train_list, y_train_list = pickle.load(f)
    f = open("./preprocessed_data/valid_list.pickle","rb")
    x_valid_list, y_valid_list = pickle.load(f)
    f = open("./preprocessed_data/train_list.pickle","rb")
    x_test_list = pickle.load(f)
    f = open("./preprocessed_data/train_y.pickle","rb")
    train_y = pickle.load(f)
    f = open("./preprocessed_data/cv.pickle","rb")
    cv = pickle.load(f)
    return x_train_list, x_valid_list, y_train_list, y_valid_list, x_test_list, train_y, cv

x_train_list, x_valid_list, y_train_list, y_valid_list, x_test_list, train_y, cv = load_preprocessed_data()


def fit_lgbm(x_train_list, x_valid_list, y_train_list, y_valid_list, y, cv, drop_columns, params: dict=None, verbose=100):

    # パラメータがないときはからの dict で置き換える
    if params is None:
        params = {}

    models = []
    oof_pred = np.zeros_like(y, dtype=np.float)
    use_threshold = 0
    ave_score = 0
    for x_train, x_valid, y_train, y_valid,(idx_train, idx_valid) in zip(x_train_list, x_valid_list, y_train_list, y_valid_list,cv): 
        x_train = x_train.drop(drop_columns,axis=1)
        x_valid = x_valid.drop(drop_columns,axis=1)
        train_data = lgb.Dataset(data=x_train, label=y_train)
        test_data = lgb.Dataset(data=x_valid, label=y_valid)
        clf = lgb.train(train_set=train_data,
            params=params,
            valid_sets=[train_data, test_data], 
            valid_names=['Train', 'Test'],
            # feval=f1_metric,
            early_stopping_rounds=100,
            verbose_eval=100
        )
        pred_i = clf.predict(x_valid)
        oof_pred[idx_valid] = pred_i
        models.append(clf)
        best_threshold = threshold_optimization(y[idx_valid], oof_pred[idx_valid], metrics=f1_score)
        score = f1_score(y[idx_valid], oof_pred[idx_valid] >= best_threshold)
        print('best_threshold: {:.4f}'.format(best_threshold))
        print('score: {:.4f}'.format(score))
        ave_score += score / 5
        use_threshold += best_threshold / 5      
        
    # best_threshold = threshold_optimization(y, oof_pred, metrics=f1_score)
    score = f1_score(y, oof_pred >= use_threshold)
    log_loss_score = log_loss(y, oof_pred)
    print('FINISHED \ logloss score: {:.4f}'.format(log_loss_score))
    print('FINISHED \ whole threshold: {:.4f}'.format(use_threshold))
    print('FINISHED \ average score: {:.4f}'.format(ave_score))
    print('FINISHED \ whole score: {:.4f}'.format(score))
    return oof_pred, models, use_threshold

# 最適な閾値を求める関数
def threshold_optimization(y_true, y_pred, metrics=None):
    def f1_opt(x):
        if metrics is not None:
            score = -metrics(y_true, y_pred >= x)
        else:
            raise NotImplementedError
        return score
    result = minimize(f1_opt, x0=np.array([0.5]), method='Nelder-Mead')
    best_threshold = result['x'].item()
    return best_threshold

def f1_metric(y_pred, data):
    y_true = data.get_label()
    score = f1_score(y_true, np.round(y_pred))
    return 'f1', score, True

params = {
    "n_estimators": 10000,
    "objective": 'binary',
    "learning_rate": 0.01,
    "num_leaves": 15,
    "random_state": 42,
    "n_jobs": -1,
    "min_data_in_leaf": 30,
}
cat_features = ['category1', 'category2', 'country', 'category3']
drop_columns = cat_features + ['html_content','goal', 'html_text']
oof_lgb, models_lgb, threshold_lgb = fit_lgbm(x_train_list, x_valid_list, y_train_list, y_valid_list, train_y, cv, drop_columns, params)

def fit_nn(x_train_list, x_valid_list, y_train_list, y_valid_list, y, cv, drop_columns):
    models = []
    oof_pred = np.zeros_like(y, dtype=np.float)
    scalers = []
    use_threshold = 0
    ave_score = 0
    historys = []
    set_seed(42)
    for x_train, x_valid, y_train, y_valid,(idx_train, idx_valid) in zip(x_train_list, x_valid_list, y_train_list, y_valid_list,cv): 
        x_train = x_train.drop(drop_columns,axis=1)#.values
        x_valid = x_valid.drop(drop_columns,axis=1)#.values
        x_train = x_train.replace(np.inf, 1)
        x_valid = x_valid.replace(np.inf, 1)
        y_train, y_valid = np_utils.to_categorical(y_train), np_utils.to_categorical(y_valid)
        scaler = StandardScaler()
        scaler.fit(x_train)
        x_train = scaler.transform(x_train)
        x_valid = scaler.transform(x_valid)
        scalers.append(scaler)
        opt = optimizers.Adam(learning_rate=0.001,epsilon=1e-08)
        model = Sequential()
        model.add(Dense(50, activation='relu', input_shape=(x_train.shape[1],)))
        model.add(Dropout(0.2))
        model.add(Dense(2, activation='sigmoid'))
        model.compile(loss='binary_crossentropy',optimizer=opt)
        batch_size = 16
        epochs = 400
        early_stopping = EarlyStopping(patience=10,verbose=1)
        history = model.fit(x_train,
                        y_train,
                        epochs= epochs,
                        batch_size= batch_size,
                        validation_data=(x_valid, y_valid),
                        callbacks=[early_stopping],
                        verbose=0)
        pred_i = model.predict(x_valid)[:,1]
        oof_pred[idx_valid] = pred_i
        historys.append(history)
        models.append(model)
        best_threshold = threshold_optimization(y[idx_valid], oof_pred[idx_valid], metrics=f1_score)
        score = f1_score(y[idx_valid], oof_pred[idx_valid] >= best_threshold)
        print('best_threshold: {:.4f}'.format(best_threshold))
        print('score: {:.4f}'.format(score))
        ave_score += score / 5
        use_threshold += best_threshold / 5
    score = f1_score(y, oof_pred >= use_threshold)
    log_loss_score = log_loss(y, oof_pred)
    print('FINISHED \ logloss score: {:.4f}'.format(log_loss_score))
    print('FINISHED \ whole threshold: {:.4f}'.format(use_threshold))
    print('FINISHED \ average score: {:.4f}'.format(ave_score))
    print('FINISHED \ whole score: {:.4f}'.format(score))
    return oof_pred, models, use_threshold, scalers

sdv_columns = ['raw_html_tfidf_sdv8_000', 'raw_html_tfidf_sdv8_001','raw_html_tfidf_sdv8_002', 'raw_html_tfidf_sdv8_003',
               'raw_html_tfidf_sdv8_004', 'raw_html_tfidf_sdv8_005','raw_html_tfidf_sdv8_006', 'raw_html_tfidf_sdv8_007',
               'removed_tags_html_tfidf_sdv8_000', 'removed_tags_html_tfidf_sdv8_001',
               'removed_tags_html_tfidf_sdv8_002', 'removed_tags_html_tfidf_sdv8_003',
               'removed_tags_html_tfidf_sdv8_004', 'removed_tags_html_tfidf_sdv8_005',
               'removed_tags_html_tfidf_sdv8_006', 'removed_tags_html_tfidf_sdv8_007',]

oof_nn, models_nn, threshold_nn, scalers = fit_nn(x_train_list, x_valid_list, y_train_list, y_valid_list, train_y, cv, drop_columns+sdv_columns)

print('emsemble')
emsemble_log_loss_score = log_loss(train_y, (oof_nn*3+oof_lgb*7)/10)
print('logloss score: {:.4f}'.format(emsemble_log_loss_score))
emsemble_f1_score = f1_score(train_y, (oof_nn*3+oof_lgb*7)/10 >= (threshold_nn*3+threshold_lgb*7)/10)
print('f1 score: {:.4f}'.format(emsemble_f1_score))

def save_models(models_lgb, threshold_lgb, models_nn, threshold_nn, scalers):
    f = open("./models/models_lgb.pickle","wb")
    pickle.dump(models_lgb, f)
    f = open("./models/threshold_lgb.pickle","wb")
    pickle.dump(threshold_lgb, f)
    for i, model_nn in enumerate(models_nn):
        model_nn.save("./models/models_nn_"+str(i)+".h5")
    f = open("./models/threshold_nn.pickle","wb")
    pickle.dump(threshold_nn, f)
    f = open("./models/scalers.pickle","wb")
    pickle.dump(scalers, f)
    
save_models(models_lgb, threshold_lgb, models_nn, threshold_nn, scalers)
