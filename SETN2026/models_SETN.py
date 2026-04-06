import pandas as pd
import seaborn as sns
import plotly.express as px
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, average_precision_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (f1_score, accuracy_score, recall_score,
                             precision_score, confusion_matrix, classification_report)
from sklearn.svm import SVC
from joblib import dump
from sklearn.metrics import precision_recall_curve, auc
from sklearn.preprocessing import label_binarize
from xgboost import XGBClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from nltk.tokenize import word_tokenize


nltk.download('stopwords')
english_stopwords = stopwords.words('english')
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')
nltk.download('wordnet')
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

class TextPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, preprocess_type='none'):
        self.preprocess_type = preprocess_type

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return [preprocess_text(text, self.preprocess_type) for text in X]
def preprocess_text(text, method='none'):
    tokens = word_tokenize(text)
    if method == 'stemming':
        tokens = [stemmer.stem(token) for token in tokens]
    elif method == 'lemmatization':
        tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return ' '.join(tokens)

tfidf_params = {
    'lowercase': [True, False],
    'max_features': [500, 1000, 2000, 3000, 5000],
    'ngram_range': [(1, 1), (1, 2)],
    'stop_words': [None, english_stopwords]
}

preprocessor_params = {
    'preprocessor__preprocess_type': ['stemming', 'lemmatization']
}

models_params = {
    'logistic_regression': {
        'model_class': LogisticRegression,
        'model_params': {
            'C': [0.001, 0.002, 0.01, 0.02],
            'penalty': ['l2'],
            'solver': ['lbfgs', 'saga'],
            'class_weight': ['balanced']
        }
    },
    'random_forest': {
        'model_class': RandomForestClassifier,
        'model_params': {
            'n_estimators': [50, 100],
            'max_depth': [5, 10, 15],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 4],
            'class_weight': ['balanced'],
            'bootstrap': [True]
        }
    },
    'svm': {
        'model_class': SVC,
        'model_params': {
            'C': [ 0.1, 1],
            'kernel': ['linear', 'rbf'],
            'gamma': ['scale', 'auto', 0.1],
            'class_weight': ['balanced'],
            'probability': [True]
        }
    },
    'nb': {
        'model_class': MultinomialNB,
        'model_params':{
            'alpha': [0.001, 0.01, 0.05, 0.1,]
        }
    },
    'xgboost': {
        'model_class': XGBClassifier, 
        'model_params': {
            'learning_rate': [0.01, 0.05],
            'max_depth': [3, 4],
            'n_estimators': [100, 150],
            'subsample': [0.7, 0.8, 1.0],
            'gamma': [0.01]
        }
    }
}


def run(X_train, y_train, X_test, y_test, model_name, num_runs):
    f1_scores = []
    auc_pr_scores = []
    all_hyperparams = []

    for i in range(num_runs):
        random_state = np.random.randint(0, 1000)
        X_train_split, X_valid, y_train_split, y_valid = train_test_split(X_train, y_train, test_size=0.2, stratify=y_train, random_state=random_state)

        tfidf = TfidfVectorizer()
        model = models_params[model_name]['model_class']()
        pipeline = Pipeline([
            ('tfidf', tfidf),
            (model_name, model)
        ])

        param_grid = {f'tfidf__{key}': values for key, values in tfidf_params.items()}
        param_grid.update({f'{model_name}__{key}': values for key, values in models_params[model_name]['model_params'].items()})
        grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='f1_macro', n_jobs=-1)
        grid_search.fit(X_train_split, y_train_split)
        best_model = grid_search.best_estimator_

        y_valid_pred = best_model.predict(X_valid)
        f1_valid = f1_score(y_valid, y_valid_pred, average='macro')
        auc_pr_valid = average_precision_score(y_valid, best_model.predict_proba(X_valid), average='macro')

        y_test_pred = best_model.predict(X_test)
        f1_test = f1_score(y_test, y_test_pred, average='macro')
        auc_pr_test = average_precision_score(y_test, best_model.predict_proba(X_test), average='macro')

        f1_scores.append(f1_test)
        auc_pr_scores.append(auc_pr_test)
        all_hyperparams.append(grid_search.best_params_)

        print(f"Run {i+1}: F1 Score = {f1_test:.4f}, AUC-PR = {auc_pr_test:.4f}")
        print(f"Best Hyperparameters: {grid_search.best_params_}\n")

    print(f"\nMean F1 Score: {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")
    print(f"Mean AUC-PR: {np.mean(auc_pr_scores):.4f} ± {np.std(auc_pr_scores):.4f}")

    return f1_scores, auc_pr_scores, all_hyperparams

def runs(X_train, y_train, X_test, y_test, model_name, num_runs):
    f1_scores = []
    auc_pr_scores = []
    all_hyperparams = []

    for i in range(num_runs):
        random_state = np.random.randint(0, 1000)
        X_train_split, X_valid, y_train_split, y_valid = train_test_split(
            X_train, y_train, test_size=0.2, stratify=y_train, random_state=random_state
        )

        preprocessor = TextPreprocessor()
        tfidf = TfidfVectorizer()
        model = models_params[model_name]['model_class']()

        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('tfidf', tfidf),
            (model_name, model)
        ])

        param_grid = {f'tfidf__{key}': values for key, values in tfidf_params.items()}
        param_grid.update(preprocessor_params) 
        param_grid.update({f'{model_name}__{key}': values for key, values in models_params[model_name]['model_params'].items()})

        grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='f1_macro', n_jobs=-1)
        grid_search.fit(X_train_split, y_train_split)
        best_model = grid_search.best_estimator_

        y_valid_pred = best_model.predict(X_valid)
        f1_valid = f1_score(y_valid, y_valid_pred, average='macro')
        auc_pr_valid = average_precision_score(y_valid, best_model.predict_proba(X_valid), average='macro')

        y_test_pred = best_model.predict(X_test)
        f1_test = f1_score(y_test, y_test_pred, average='macro')
        auc_pr_test = average_precision_score(y_test, best_model.predict_proba(X_test), average='macro')

        f1_scores.append(f1_test)
        auc_pr_scores.append(auc_pr_test)
        all_hyperparams.append(grid_search.best_params_)

        print(f"Run {i+1}: F1 Score = {f1_test:.4f}, AUC-PR = {auc_pr_test:.4f}")
        print(f"Best Hyperparameters: {grid_search.best_params_}\n")

    print(f"\nMean F1 Score: {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")
    print(f"Mean AUC-PR: {np.mean(auc_pr_scores):.4f} ± {np.std(auc_pr_scores):.4f}")

    return f1_scores, auc_pr_scores, all_hyperparams


def compute_tfidf_level(train_df, test_df, column, max_features=5000, ngram_range=(1, 1), preprocess_type='none'):
    preprocessor = TextPreprocessor(preprocess_type)
    
    processed_train_text = preprocessor.transform(train_df[column])
    processed_test_text = preprocessor.transform(test_df[column])

    processed_train_text_no_stopwords = [' '.join([word for word in text.split() if word not in english_stopwords]) 
                                         for text in processed_train_text]
    processed_test_text_no_stopwords = [' '.join([word for word in text.split() if word not in english_stopwords]) 
                                        for text in processed_test_text]

    tfidf_vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=ngram_range,
        max_features=max_features,
        stop_words=None
    )

    tfidf_train_matrix = tfidf_vectorizer.fit_transform(processed_train_text_no_stopwords)
    tfidf_train_df = pd.DataFrame(tfidf_train_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out())

    tfidf_test_matrix = tfidf_vectorizer.transform(processed_test_text_no_stopwords)
    tfidf_test_df = pd.DataFrame(tfidf_test_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out())

    return tfidf_vectorizer, tfidf_train_df, tfidf_test_df


def best_model(X_train_tfidf, y_train, X_test_tfidf, y_test):
    svm = SVC(
        C=1,
        class_weight='balanced',
        gamma='scale',
        kernel='linear',
        probability=True
    )

    svm.fit(X_train_tfidf, y_train)

    y_pred = svm.predict(X_test_tfidf)
    y_proba = svm.predict_proba(X_test_tfidf)  

    f1 = f1_score(y_test, y_pred, average='macro')

    unique_classes = np.unique(y_train)
    y_test_bin = label_binarize(y_test, classes=unique_classes)

    auc_pr = np.mean([
        average_precision_score(y_test_bin[:, i], y_proba[:, i])
        for i in range(len(unique_classes))
    ])
    return f1, auc_pr


# def run_svm_model(X_train, y_train, X_test, y_test):
#     preprocessor = TextPreprocessor('stemming')
#     pipeline = Pipeline([
#         ('preprocessor', preprocessor),
#         ('tfidf', TfidfVectorizer(
#             lowercase=True,
#             max_features=2000,
#             ngram_range=(1, 1),
#             stop_words= english_stopwords
#         )),
#         ('svm', SVC(
#             C=1,
#             class_weight='balanced',
#             gamma='scale',
#             kernel='linear',
#             probability=True
#         ))
#     ])

#     pipeline.fit(X_train, y_train)

#     y_pred = pipeline.predict(X_test)
#     y_proba = pipeline.predict_proba(X_test)

#     f1 = f1_score(y_test, y_pred, average='macro')
#     auc_pr = average_precision_score(y_test, y_proba, average='macro')

#     return f1, auc_pr


def best_model_length(X_train, y_train, X_test, y_test):
    lr = LogisticRegression(
        C=0.02,
        class_weight='balanced',
        penalty='l2',
        solver='lbfgs',
    )

    # Fit the model
    lr.fit(X_train, y_train)

    y_pred = lr.predict(X_test)
    y_proba = lr.predict_proba(X_test)

    f1 = f1_score(y_test, y_pred, average='macro')

    unique_classes = np.unique(y_train)
    y_test_bin = label_binarize(y_test, classes=unique_classes)

    auc_pr = np.mean([
        average_precision_score(y_test_bin[:, i], y_proba[:, i])
        for i in range(len(unique_classes))
    ])
    
    return f1, auc_pr


def best_model_balanced(X_train_tfidf, y_train, X_test_tfidf, y_test):
    svm = SVC(
        C=1,
        class_weight='balanced',
        gamma='scale',
        kernel='rbf',
        probability=True
    )

    svm.fit(X_train_tfidf, y_train)

    y_pred = svm.predict(X_test_tfidf)
    y_proba = svm.predict_proba(X_test_tfidf)  

    f1 = f1_score(y_test, y_pred, average='macro')

    unique_classes = np.unique(y_train)
    y_test_bin = label_binarize(y_test, classes=unique_classes)

    auc_pr = np.mean([
        average_precision_score(y_test_bin[:, i], y_proba[:, i])
        for i in range(len(unique_classes))
    ])
    return f1, auc_pr

























