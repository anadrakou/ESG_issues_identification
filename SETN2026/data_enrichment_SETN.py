import pandas as pd
import torch
from transformers import MarianMTModel, MarianTokenizer, BertTokenizer, BertForSequenceClassification, pipeline, BertModel, AutoTokenizer, AutoModelForSequenceClassification, AutoModel
import transformers
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import nltk
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.preprocessing import FunctionTransformer
import random
import torch.nn.functional as F


#################################
### functions for translation ###
#################################


#model_name_fr = "Helsinki-NLP/opus-mt-fr-en"
#tokenizer_fr = MarianTokenizer.from_pretrained(model_name_fr)
#model_fr = MarianMTModel.from_pretrained(model_name_fr)

#model_name_jp = "Helsinki-NLP/opus-mt-ja-en"
#tokenizer_jp = MarianTokenizer.from_pretrained(model_name_jp)
#model_jp = MarianMTModel.from_pretrained(model_name_jp)


# def translate_french(text):
#     inputs = tokenizer_fr(text, return_tensors="pt", padding=True, truncation=True)
#     translated = model_fr.generate(**inputs)
#     return tokenizer_fr.decode(translated[0], skip_special_tokens=True)

# def translate_japanese(text):
#     inputs = tokenizer_jp(text, return_tensors="pt", padding=True, truncation=True)
#     translated = model_jp.generate(**inputs)
#     return tokenizer_jp.decode(translated[0], skip_special_tokens=True)


#########################################
### functions for scoring ESG factors ###
#########################################

finbert = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-esg', num_labels=4)
finbert.eval()
tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-esg')

nlp = pipeline("text-classification", model=finbert, tokenizer=tokenizer, return_all_scores=True)

def get_labels_and_scores(text):
    result = nlp(text)
    labels = [item['label'] for item in result[0]]
    scores = [item['score'] for item in result[0]]
    return labels, scores


def assign_scores(row):
    labels, scores = row['finbert_labels'], row['finbert_scores']

    scores_dict = {'None': 0, 'Environment': 0, 'Social': 0, 'Governance': 0}

    if len(scores) == 4:
        scores_dict['None'] = scores[0]
        scores_dict['Environment'] = scores[1]
        scores_dict['Social'] = scores[2]
        scores_dict['Governance'] = scores[3]

    return pd.Series(scores_dict)


#######################
### 768 dimensions  ###
#######################

tokenizer_esg = BertTokenizer.from_pretrained('yiyanghkust/finbert-esg')
model_esg = BertModel.from_pretrained('yiyanghkust/finbert-esg')
model_esg.eval()  

def get_bert_embeddings(text):
    inputs = tokenizer_esg(text, return_tensors='pt', truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model_esg(**inputs)
    cls_embedding = outputs.last_hidden_state[:, 0, :]  

    return cls_embedding.squeeze().numpy()


########################################
### functions for sentiment analysis ###
########################################

tokenizer_sent = AutoTokenizer.from_pretrained("descartes100/distilBERT_ESG")
sentbert = AutoModelForSequenceClassification.from_pretrained("descartes100/distilBERT_ESG")
sentbert.eval()

labels_map = [
    'Environmental_Negative', 'Environmental_Neutral', 'Environmental_Positive',
    'Social_Negative', 'Social_Neutral', 'Social_Positive',
    'Governance_Negative', 'Governance_Neutral', 'Governance_Positive'
]

def get_sentiment_probs(text):
    inputs = tokenizer_sent(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        logits = sentbert(**inputs).logits
        probs = torch.sigmoid(logits).squeeze().numpy()  
    return {labels_map[i]: probs[i] for i in range(len(probs))}

def aggregate_sentiments(df):
    df["pos"] = df[["Environmental_Positive", "Social_Positive", "Governance_Positive"]].sum(axis=1)
    df["neu"] = df[["Environmental_Neutral", "Social_Neutral", "Governance_Neutral"]].sum(axis=1)
    df["neg"] = df[["Environmental_Negative", "Social_Negative", "Governance_Negative"]].sum(axis=1)
    return df

#######################
### 768 dimensions  ###
#######################

tokenizer_sent = AutoTokenizer.from_pretrained("descartes100/distilBERT_ESG")
model_sent = AutoModel.from_pretrained("descartes100/distilBERT_ESG")
model_sent.eval()

def get_sent_embeddings(text):
    inputs = tokenizer_sent(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model_sent(**inputs)
    cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
    return cls_embedding



#############################
### FR + EN data balance  ###
#############################

def balance_two_columns_by_undersample(
    df,
    col_a,
    col_b,
    strategy='median',      # 'median' | 'mean' | 'min' | 'fixed'
    fixed_n=None,           # used when strategy == 'fixed'
    random_state=42,
    verbose=True
):
    """
    Undersample rows so that each joint group (col_a, col_b) has at most `cap` rows.
    Cap determination:
      - 'median' : cap = median group size
      - 'mean'   : cap = int(mean group size)
      - 'min'    : cap = min group size (aggressive)
      - 'fixed'  : cap = fixed_n (must provide fixed_n)
    Rows in groups with size <= cap are kept unchanged.
    Returns the undersampled dataframe (shuffled).
    """
    # compute joint counts
    group_counts = df.groupby([col_a, col_b]).size().reset_index(name='count')
    counts = group_counts['count']
    
    if strategy == 'median':
        cap = int(counts.median())
    elif strategy == 'mean':
        cap = int(counts.mean())
    elif strategy == 'min':
        cap = int(counts.min())
    elif strategy == 'fixed':
        if fixed_n is None:
            raise ValueError("fixed_n must be provided when strategy='fixed'")
        cap = int(fixed_n)
    else:
        raise ValueError("strategy must be one of 'median','mean','min','fixed'")
    
    if cap < 1:
        cap = 1

    if verbose:
        print(f"Balancing joint groups ({col_a}, {col_b}) with cap = {cap} (strategy={strategy})")
        print("Group counts before (sample):")
        print(group_counts.sort_values('count', ascending=False).head(10).to_string(index=False))
    
    sampled_parts = []
    grouped = df.groupby([col_a, col_b])
    for (a_val, b_val), group in grouped:
        n = len(group)
        if n <= cap:
            sampled_parts.append(group)
        else:
            sampled = group.sample(n=cap, random_state=random_state)
            sampled_parts.append(sampled)
    
    balanced_df = pd.concat(sampled_parts, axis=0).reset_index(drop=True)
    balanced_df = balanced_df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    
    if verbose:
        print("\nCounts after balancing (joint groups):")
        after_counts = balanced_df.groupby([col_a, col_b]).size().reset_index(name='count')
        print(after_counts.sort_values('count', ascending=False).head(10).to_string(index=False))
        # marginal summaries
        print("\nMarginal counts after balancing:")
        print(balanced_df[col_a].value_counts().to_string())
        print(balanced_df[col_b].value_counts().to_string())
    
    return balanced_df