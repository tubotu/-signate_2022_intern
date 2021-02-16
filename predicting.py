import os
import random
import re
import warnings
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler
from keras.models import load_model
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.random import set_seed
import pickle

warnings.filterwarnings("ignore")

INPUT = "./input"
SUBMISSION = "./submission"
FOLDS = 5  # kfoldの数

def load_models_and_test():
    f = open("./models/models_lgb.pickle","rb")
    models_lgb = pickle.load(f)
    f = open("./models/threshold_lgb.pickle","rb")
    threshold_lgb = pickle.load(f)
    models_nn = [load_model("./models/models_nn_"+str(i)+".h5") for i in range(FOLDS)]
    f = open("./models/threshold_nn.pickle","rb")
    threshold_nn = pickle.load(f)
    f = open("./models/scalers.pickle","rb")
    scalers = pickle.load(f)
    f = open("./preprocessed_data/test_list.pickle","rb")
    x_test_list = pickle.load(f)
    return models_lgb, threshold_lgb, models_nn, threshold_nn, scalers, x_test_list

models_lgb, threshold_lgb, models_nn, threshold_nn, scalers, x_test_list = load_models_and_test()

cat_features = ['category1', 'category2', 'country', 'category3']
drop_columns = cat_features + ['html_content','goal', 'html_text']
sdv_columns = ['raw_html_tfidf_sdv8_000', 'raw_html_tfidf_sdv8_001','raw_html_tfidf_sdv8_002', 'raw_html_tfidf_sdv8_003',
               'raw_html_tfidf_sdv8_004', 'raw_html_tfidf_sdv8_005','raw_html_tfidf_sdv8_006', 'raw_html_tfidf_sdv8_007',
               'removed_tags_html_tfidf_sdv8_000', 'removed_tags_html_tfidf_sdv8_001',
               'removed_tags_html_tfidf_sdv8_002', 'removed_tags_html_tfidf_sdv8_003',
               'removed_tags_html_tfidf_sdv8_004', 'removed_tags_html_tfidf_sdv8_005',
               'removed_tags_html_tfidf_sdv8_006', 'removed_tags_html_tfidf_sdv8_007',]

test_pred_nn = np.mean(np.array([model.predict(scaler.transform(test_X.drop(drop_columns+sdv_columns,axis=1).replace(np.inf, 1)))[:,1] 
                                 for test_X, model, scaler in zip(x_test_list, models_nn, scalers)]),axis=0)
test_pred_lgb = np.mean(np.array([model.predict(test_X.drop(drop_columns,axis=1)) 
                                  for test_X, model in zip(x_test_list, models_lgb)]), axis=0)

# submit
test_pred = (test_pred_nn*3+test_pred_lgb*7)/10
sub = pd.read_csv(os.path.join(INPUT, "sample_submit.csv"), header=None, index_col=0)
sub[1] = (test_pred >= (threshold_nn*3+threshold_lgb*7)/10).astype(int)
sub.to_csv(os.path.join(SUBMISSION, "submission.csv"))
