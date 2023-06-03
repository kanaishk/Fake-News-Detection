import os
import pandas as pd
import numpy as np
import random
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report

print('Start')
cwd = os.getcwd()
dataset_dir = os.path.join(cwd,'Dataset')
result_dir = os.path.join(cwd,'Results')
model_dir = os.path.join(cwd,'Model')
df = pd.read_csv(os.path.join(dataset_dir,'train_news_preprocessed.csv'), low_memory=False, 
                 usecols = ['label','clean_news_tokens'])
df_bf = pd.read_csv(os.path.join(dataset_dir,'train_news_preprocessed_bf.csv'), low_memory=False, 
                 usecols = ['label','clean_news_tokens'])

RANDOM_STATE = 1973
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
pd.options.display.max_seq_items = 20
pd.options.display.max_rows = 50

print('Tokens')
X = df['clean_news_tokens'].apply(lambda x: ' '.join(eval(x)))  # Concatenate tokens into space-separated strings
y = df['label']  # Target variable (fake or not fake)
X_valid_bf = df_bf['clean_news_tokens'].apply(lambda x: ' '.join(eval(x))) 
y_valid_bf = df_bf['label']

print('Splitter')
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

print('TFIDF')
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train).copy()
#X_valid_tfidf = vectorizer.transform(X_valid).copy()
X_valid_tfidf_bf = vectorizer.transform(X_valid_bf).copy()

modelname = 'Voting'
print('Model Training')

with open(os.path.join(model_dir,'trained_base_model.pkl'), 'rb') as f:
    best_estimator = pickle.load(f)

'''
y_pred_valid = best_estimator.predict(X_valid_tfidf)

report = classification_report(y_valid, y_pred_valid, digits=5)
print(report)
with open(os.path.join(result_dir, f'BASE/BASE_{modelname}_Valid_report.txt'), 'w') as f:
    f.write(report)

y_pred_valid_mc = best_estimator.predict(X_valid_tfidf_mc)

report = classification_report(y_valid_mc, y_pred_valid_mc, digits=5)
print(report)
with open(os.path.join(result_dir, f'BASE/BASE_{modelname}_Valid_report_mc.txt'), 'w') as f:
    f.write(report)
'''
y_pred_valid_bf = best_estimator.predict(X_valid_tfidf_bf)

report = classification_report(y_valid_bf, y_pred_valid_bf, digits=5)
print(report)
with open(os.path.join(result_dir, f'BASE/BASE_{modelname}_Valid_report_bf.txt'), 'w') as f:
    f.write(report)
