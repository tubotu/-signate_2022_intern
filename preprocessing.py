#!/usr/bin/env python
# coding: utf-8
import os
import random
import re
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import NMF, PCA, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import f1_score, log_loss
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
import category_encoders as ce
import nlp
import texthero as hero
import xfeat
from xfeat import TargetEncoder
import pickle

warnings.filterwarnings("ignore")

INPUT = "./input"
SUBMISSION = "./submission"
NAME = "baseline001"
BERT_PRED = './bert_pred'
FOLDS = 5  # kfoldの数

train = pd.read_csv(os.path.join(INPUT, "train.csv"))
test = pd.read_csv(os.path.join(INPUT, "test.csv"))

def goal2feature(input_df):
    tmp = input_df.copy()
    tmp = tmp["goal"].replace("100000+", "100000-101000")
    tmp = np.array([g.split("-")[1] for g in tmp], dtype="int")
    output_df = pd.DataFrame(tmp, columns=["goal_max"])
    return pd.concat([input_df,output_df],axis=1)

train = goal2feature(train)
test = goal2feature(test)

def agg_category2(train, test):
    tmp = train.append(test).reset_index(drop=True)
    tmp['goal_max_mul_duration'] = tmp['goal_max'] * tmp['duration']
    group_key = "category2"
    group_values = [
        "goal_max_mul_duration",
    ]
    agg_methods = ["std"]
    output_df, cols = xfeat.aggregation(tmp, group_key, group_values, agg_methods)
    output_train = output_df.iloc[:len(train)]
    output_test = output_df.iloc[len(train):].reset_index(drop=True).drop('state',axis=1)
    return output_train, output_test

def make_combination_feature(input_df):
    tmp = input_df.copy()
    tmp["category3"] = tmp["category1"] + tmp["category2"] 
    return tmp

train = make_combination_feature(train)
test = make_combination_feature(test)
train, test = agg_category2(train, test)

def text_vectorizer(input_df, 
                    text_columns,
                    cleansing_hero=None,
                    vectorizer=CountVectorizer(),
                    transformer=TruncatedSVD(n_components=128),
                    name='html_count_svd'):
    
    output_df = pd.DataFrame()
    output_df[text_columns] = input_df[text_columns].astype(str).fillna('missing')
    features = []
    for c in text_columns:
        if cleansing_hero is not None:
            output_df[c] = cleansing_hero(output_df, c)

        sentence = vectorizer.fit_transform(output_df[c])
        feature = transformer.fit_transform(sentence)
        num_p = feature.shape[1]
        feature = pd.DataFrame(feature, columns=[name+str(num_p) + f'_{i:03}' for i in range(num_p)])
        features.append(feature)
    output_df = pd.concat(features, axis=1)
    return output_df

def get_text_vector_raw__tfidf_sdv64(train, test):
    tmp = train.append(test).reset_index(drop=True)
    output_df = text_vectorizer(tmp,
                                ["html_content"],
                                cleansing_hero=None,
                                vectorizer=TfidfVectorizer(),
                                transformer=TruncatedSVD(n_components=8, random_state=2021),
                                name="raw_html_tfidf_sdv"
                                )
    output_df = pd.concat([tmp,output_df], axis=1)
    output_train = output_df.iloc[:len(train)]
    output_test = output_df.iloc[len(train):].reset_index(drop=True).drop('state',axis=1)
    return output_train, output_test

def cleansing_hero_remove_html_tags(input_df, text_col):
    ## only remove html tags, do not remove punctuation
    custom_pipeline = [
        hero.preprocessing.fillna,
        hero.preprocessing.remove_html_tags,
        hero.preprocessing.lowercase,
        hero.preprocessing.remove_digits,
        hero.preprocessing.remove_stopwords,
        hero.preprocessing.remove_whitespace,
        hero.preprocessing.stem
    ]
    texts = hero.clean(input_df[text_col], custom_pipeline)
    return texts

# html tag除去 html_content [tfidf -> sdv で次元削減(64)]
def get_text_vector_removed_html_tags__tfidf_sdv64(train, test):
    tmp = train.append(test).reset_index(drop=True)
    output_df = text_vectorizer(tmp,
                                ["html_content"],
                                cleansing_hero=cleansing_hero_remove_html_tags,
                                vectorizer=TfidfVectorizer(),
                                transformer=TruncatedSVD(n_components=8, random_state=2021),
                                name="removed_tags_html_tfidf_sdv"
                                )
    output_df = pd.concat([tmp,output_df], axis=1)
    output_train = output_df.iloc[:len(train)]
    output_test = output_df.iloc[len(train):].reset_index(drop=True).drop('state',axis=1)
    return output_train, output_test

train, test = get_text_vector_raw__tfidf_sdv64(train, test)
train, test = get_text_vector_removed_html_tags__tfidf_sdv64(train, test)

def add_count_features(train, test, col_names, target_col_name):
    df = train.append(test).reset_index(drop=True)
    for col_name in col_names:
        counts = df.groupby(col_name)[target_col_name].count()
        train[f"{col_name}_counts"] = train[col_name].map(counts)
        test[f"{col_name}_counts"] = test[col_name].map(counts)
    return train, test

cat_features = ['category1', 'category2', 'country', 'category3']
train, test = add_count_features(train, test, cat_features, 'state')

def cleansing_hero_remove_html_tags(input_df, text_col):
    ## only remove html tags, do not remove punctuation
    custom_pipeline = [
        hero.preprocessing.fillna,
        hero.preprocessing.remove_html_tags,
        hero.preprocessing.remove_whitespace,
    ]
    texts = hero.clean(input_df[text_col], custom_pipeline)
    return texts

train['html_text'] = cleansing_hero_remove_html_tags(train, 'html_content')
test['html_text'] = cleansing_hero_remove_html_tags(test, 'html_content')

def bert_feature_add(train_df, test_df):
    train = train_df.copy()
    test = test_df.copy()
    train['bert_pred'] = np.load(os.path.join(BERT_PRED, 'oof_pred_bert.npy'))
    test['bert_pred'] = np.load(os.path.join(BERT_PRED,'test_pred_bert.npy'))
    train['roberta_pred'] = np.load(os.path.join(BERT_PRED, 'oof_pred_roberta.npy'))
    test['roberta_pred'] = np.load(os.path.join(BERT_PRED, 'test_pred_roberta.npy'))
    train['bert_pred_512'] = np.load(os.path.join(BERT_PRED, 'oof_pred_bert_512.npy'))
    test['bert_pred_512']= np.load(os.path.join(BERT_PRED, 'test_pred_bert_512.npy'))
    train['roberta_pred_512'] = np.load(os.path.join(BERT_PRED, 'oof_pred_roberta_512.npy'))
    test['roberta_pred_512'] = np.load(os.path.join(BERT_PRED, 'test_pred_roberta_512.npy'))
    train['bert_pred_emsemble'] = (train['bert_pred'] + train['roberta_pred'] + train['bert_pred_512'] + train['roberta_pred_512'])/4
    test['bert_pred_emsemble'] = (train['bert_pred'] + train['roberta_pred'] + train['bert_pred_512'] + train['roberta_pred_512'])/4
    return train, test

train, test = bert_feature_add(train, test)

def html_feature(df):
    tmp = df.copy()
    tmp['text_words'] = tmp['html_text'].str.count(' ')
    tmp['html_img'] = tmp['html_content'].str.count('</figure>')
    tmp['html_video'] = tmp['html_content'].str.count('</video>')
    tmp['bold'] = tmp['html_content'].str.count('bold')
    tmp['html_img_video'] = tmp['html_img'] + tmp['html_video']
    tmp['img_video_div_text'] = tmp['html_img_video'] / tmp['text_words']
    return tmp

train = html_feature(train)
test = html_feature(test)

# 説明変数と目的変数を分ける。
train_y = train["state"].copy()
train_X = train.drop(['id','state'], axis=1).copy()
test_X = test.drop(['id'], axis=1)
skfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=2020)
cv = list(skfold.split(train_X, train_y))

def target_encoding_train(x_train, y_train,input_cols):
    train = pd.concat([x_train,y_train],axis=1)
    Kfold = KFold(n_splits=5, shuffle=True, random_state=2021)
    encoder = TargetEncoder(
        input_cols=input_cols,
        target_col=y_train.name,
        fold = Kfold,
        output_suffix="_te"
    )
    train = encoder.fit_transform(train)
    return train.drop([y_train.name],axis=1),train[y_train.name]

def target_encoding_valid(x_train, y_train, x_valid, input_cols):
    for col in input_cols:
        data_tmp = pd.DataFrame({col: x_train[col], 'target': y_train})
        target_mean = data_tmp.groupby(col)['target'].mean()
        x_valid.loc[:,col+'_te'] = x_valid[col].map(target_mean)
    return x_valid

def target_encoding_cv(train_X, train_y, test_X,cat_features, cv):
    x_train_list = []
    x_valid_list = []
    y_train_list = []
    y_valid_list = []
    x_test_list = []
    for (idx_train, idx_valid) in cv:
        # training data を train/valid に分割
        x_train, y_train = train_X.iloc[idx_train], train_y.iloc[idx_train]
        x_valid, y_valid = train_X.iloc[idx_valid], train_y.iloc[idx_valid]
        x_valid = target_encoding_valid(x_train, y_train, x_valid, cat_features)
        x_train, y_train = target_encoding_train(x_train, y_train, cat_features)
        x_test = target_encoding_valid(x_train, y_train, test_X, cat_features)
        x_train = x_train.fillna(x_train.median())
        x_valid = x_valid.fillna(x_valid.median())
        x_test = x_test.fillna(x_test.median())
        x_train_list.append(x_train)
        x_valid_list.append(x_valid)
        y_train_list.append(y_train)
        y_valid_list.append(y_valid)
        x_test_list.append(x_test)
    return x_train_list, x_valid_list, y_train_list, y_valid_list, x_test_list

x_train_list, x_valid_list, y_train_list, y_valid_list, x_test_list = target_encoding_cv(train_X, train_y, test_X, cat_features, cv)

def save_preprocessed_data(x_train_list, x_valid_list, y_train_list, y_valid_list, x_test_list, train_y, cv):
    f = open("./preprocessed_data/train_list.pickle","wb")
    pickle.dump([x_train_list, y_train_list], f)
    f = open("./preprocessed_data/valid_list.pickle","wb")
    pickle.dump([x_valid_list, y_valid_list], f)
    f = open("./preprocessed_data/test_list.pickle","wb")
    pickle.dump(x_test_list, f)
    f = open("./preprocessed_data/train_y.pickle","wb")
    pickle.dump(train_y, f)
    f = open("./preprocessed_data/cv.pickle","wb")
    pickle.dump(cv, f)

save_preprocessed_data(x_train_list, x_valid_list, y_train_list, y_valid_list, x_test_list, train_y, cv)
