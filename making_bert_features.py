import collections
import os
import random
import re
import warnings

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy.optimize import minimize
from sklearn.decomposition import NMF, PCA, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold
from tqdm.notebook import tqdm
from transformers import AdamW, AutoConfig, AutoModel, AutoTokenizer, BertModel

import category_encoders as ce
import nlp
import texthero as hero
import xfeat
from xfeat import TargetEncoder

warnings.filterwarnings("ignore")

INPUT = "./input"
FOLDS = 5  # kfoldの数

train = pd.read_csv(os.path.join(INPUT, "train.csv"))
test = pd.read_csv(os.path.join(INPUT, "test.csv"))

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

# 説明変数と目的変数を分ける。
train_y = train["state"].copy()
train_X = train.drop(['id','state'], axis=1).copy()

test_X = test.drop(['id'], axis=1)

skfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=2020)
cv = list(skfold.split(train_X, train_y))

SEED = 42
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
seed_everything(SEED)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
MODELS_DIR = "./models_bert/"
MODEL_NAME = 'roberta-base'
TRAIN_BATCH_SIZE = 16
VALID_BATCH_SIZE = 16
NUM_CLASSES = 2
EPOCHS = 5
NUM_SPLITS = 5

def make_folded_df(df, cv):
    df["kfold"] = np.nan
    df = df.rename(columns={'state': 'labels'})
    label = df["labels"].tolist()
    for fold, (_, valid_indexes) in enumerate(cv):
        df.iloc[valid_indexes, df.columns.get_loc('kfold')] = fold
    return df

def make_dataset(df, tokenizer, device):
    dataset = nlp.Dataset.from_pandas(df)
    dataset = dataset.map(
        lambda example: tokenizer(example["html_text"],
                                  padding="max_length",
                                  truncation=True,
                                  max_length=256))
    dataset.set_format(type='torch', 
                       columns=['input_ids'
                                #, 'token_type_ids'
                                , 'attention_mask'
                                , 'labels'], 
                       device=device)
    return dataset

class Classifier(nn.Module):
    def __init__(self, model_name, num_classes=2):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name, return_dict=False)
        # config = AutoConfig.from_pretrained(model_name)
        # self.bert = BertModel.from_pretrained(model_name, return_dict=False) 
        self.dropout = nn.Dropout(0.1)
        self.linear = nn.Linear(768, num_classes)
        nn.init.normal_(self.linear.weight, std=0.03)
        nn.init.ones_(self.linear.bias)
        #nn.init.zeros_(self.linear.bias)

    def forward(self, input_ids, attention_mask):
        output, _ = self.bert(
            input_ids = input_ids,
            attention_mask = attention_mask,
            #token_type_ids = token_type_ids
        )
        output = output[:, 0, :]
        output = self.dropout(output)
        output = self.linear(output)
        return output

models = []
for fold in range(NUM_SPLITS):
    model = Classifier(MODEL_NAME)
    model.load_state_dict(torch.load(MODELS_DIR + f"best_{MODEL_NAME}_{fold}.pth"))
    model.to(DEVICE)
    model.eval()
    models.append(model)

def bert_pred_cv(train_df, models, cv):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train = train_df.copy()
    train['labels'] = -1
    oof_pred = np.zeros_like(train['labels'], dtype=np.float)
    with torch.no_grad():
        for model, (_, idx_valid)  in zip(models, cv):
            valid = train.iloc[idx_valid]
            valid_dataset = make_dataset(valid, tokenizer, DEVICE)
            valid_dataloader = torch.utils.data.DataLoader(
                valid_dataset, batch_size=VALID_BATCH_SIZE, shuffle=False)
            progress = tqdm(valid_dataloader, total=len(valid_dataloader))
            valid_output = []
            for batch in progress:
                progress.set_description("<Valid>")
                attention_mask, input_ids, labels = batch.values()
                output = model(input_ids, attention_mask)
                output = np.array(torch.softmax(output, dim=1).cpu().detach())[:,1]
                valid_output.extend(output)
            oof_pred[idx_valid] = valid_output
        score = f1_score(train['state'], np.round(oof_pred))
        print('FINISHED \ whole score: {:.4f}'.format(score))
    return oof_pred

oof_pred_roberta = bert_pred_cv(train, models, cv)

def bert_pred_test(test_df, models):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    test = test_df.copy()
    test['labels'] = -1
    test_pred = np.zeros_like(test['labels'], dtype=np.float)
    test_dataset = make_dataset(test, tokenizer, DEVICE)
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=VALID_BATCH_SIZE, shuffle=False)
    with torch.no_grad():
        progress = tqdm(test_dataloader, total=len(test_dataloader))
        final_output = []

        for batch in progress:
            progress.set_description("<Test>")

            attention_mask, input_ids, labels = batch.values()

            outputs = []
            for model in models:
                output = model(input_ids, attention_mask)
                outputs.append(output)

            outputs = sum(outputs) / len(outputs)
            outputs = np.array(torch.softmax(outputs, dim=1).cpu().detach())[:,1]

            final_output.extend(outputs)

    return final_output

test_pred_roberta = bert_pred_test(test, models)

np.save('./bert_pred/oof_pred', oof_pred_roberta)
np.save('./bert_pred/test_pred', test_pred_roberta)

