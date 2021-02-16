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

def train_fn(dataloader, model, criterion, optimizer, scheduler, device, epoch):
    model.train()
    # 勾配計算を最後のBertLayerモジュールと追加した分類アダプターのみ実行
    
    total_loss = 0
    total_corrects = 0
    all_labels = []
    all_preds = []

    progress = tqdm(dataloader, total=len(dataloader))

    for i, batch in enumerate(progress):
        progress.set_description(f"<Train> Epoch{epoch+1}")

        attention_mask, input_ids, labels = batch.values()
        del batch

        optimizer.zero_grad()

        outputs = model(input_ids, attention_mask)
        del input_ids, attention_mask
        loss = criterion(outputs, labels)  # 損失を計算
        _, preds = torch.max(outputs, 1)  # ラベルを予測
        del outputs

        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        del loss
        total_corrects += torch.sum(preds == labels)

        all_labels += labels.tolist()
        all_preds += preds.tolist()
        del labels, preds

        progress.set_postfix(loss=total_loss/(i+1), f1=f1_score(all_labels, all_preds, average="macro"))

    train_loss = total_loss / len(dataloader)
    train_acc = total_corrects.double().cpu().detach().numpy() / len(dataloader.dataset)
    train_f1 = f1_score(all_labels, all_preds, average="macro")

    return train_loss, train_acc, train_f1

def eval_fn(dataloader, model, criterion, device, epoch):
    model.eval()
    total_loss = 0
    total_corrects = 0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        progress = tqdm(dataloader, total=len(dataloader))
        
        for i, batch in enumerate(progress):
            progress.set_description(f"<Valid> Epoch{epoch+1}")

            attention_mask, input_ids, labels = batch.values()
            del batch

            outputs = model(input_ids, attention_mask)
            del input_ids, attention_mask
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            del outputs

            total_loss += loss.item()
            del loss
            total_corrects += torch.sum(preds == labels)

            all_labels += labels.tolist()
            all_preds += preds.tolist()
            del labels, preds

            progress.set_postfix(loss=total_loss/(i+1), f1=f1_score(all_labels, all_preds, average="macro"))

    valid_loss = total_loss / len(dataloader)
    valid_acc = total_corrects.double().cpu().detach().numpy() / len(dataloader.dataset)

    valid_f1 = f1_score(all_labels, all_preds, average="macro")

    return valid_loss, valid_acc, valid_f1

def trainer(fold, df):

    train_df = df[df.kfold != fold].reset_index(drop=True)
    valid_df = df[df.kfold == fold].reset_index(drop=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_dataset = make_dataset(train_df, tokenizer, DEVICE)
    valid_dataset = make_dataset(valid_df, tokenizer, DEVICE)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True
    )
    valid_dataloader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=VALID_BATCH_SIZE, shuffle=False
    )

    model = Classifier(MODEL_NAME, num_classes=NUM_CLASSES)
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=2e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100000, gamma=1.0)
    # ダミーのスケジューラー

    train_losses = []
    train_accs = []
    train_f1s = []
    valid_losses = []
    valid_accs = []
    valid_f1s = []

    best_loss = np.inf
    best_acc = 0
    best_f1 = 0
    for epoch in range(EPOCHS):
        train_loss, train_acc, train_f1 = train_fn(train_dataloader, model, criterion, optimizer, scheduler, DEVICE, epoch)
        valid_loss, valid_acc, valid_f1 = eval_fn(valid_dataloader, model, criterion, DEVICE, epoch)
        print(f"Loss: {valid_loss}  Acc: {valid_acc}  f1: {valid_f1}  ", end="")

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        train_f1s.append(train_f1)
        valid_losses.append(valid_loss)
        valid_accs.append(valid_acc)
        valid_f1s.append(valid_f1)
        
        best_acc = valid_acc if valid_acc > best_acc else best_acc
        best_f1 = valid_f1 if valid_f1 > best_f1 else best_f1
        if valid_loss < best_loss:
            best_loss = valid_loss
            print("model saving!", end="")
            torch.save(model.state_dict(), MODELS_DIR + f"best_{MODEL_NAME}_{fold}.pth")
        else:
            break
        print("\n")

    return best_f1

df = make_folded_df(train[['state', 'html_text']], cv)
f1_scores = []
for fold in range(NUM_SPLITS):
    print(f"fold {fold}", "="*80)
    f1 = trainer(fold, df)
    f1_scores.append(f1)
    print(f"<fold={fold}> best score: {f1}\n")
