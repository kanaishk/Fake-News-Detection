import os
import random
import pandas as pd
import numpy as np 
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import make_scorer, accuracy_score, f1_score, recall_score, precision_score
from sklearn.metrics import classification_report

cwd = os.getcwd()
dataset_dir = os.path.join(cwd,'Dataset')
result_dir = os.path.join(cwd,'Results')
df = pd.read_csv(os.path.join(dataset_dir,'train_news_preprocessed.csv'), low_memory=False, 
                 usecols = ['label','clean_news_tokens','clean_headline_tokens'])
# 'headline','news','headline_len','news_len','caps_in_headline','caps_in_news',

RANDOM_STATE = 1973
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
pd.options.display.max_seq_items = 20
pd.options.display.max_rows = 50

X = df['clean_news_tokens'].apply(lambda x: ' '.join(eval(x)))  # Concatenate tokens into space-separated strings
y = df['label']  # Target variable (fake or not fake)

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_valid_tfidf = vectorizer.transform(X_valid)

Iters = 100
parallel_workers = 8
cross_val_works = 5
verbose = 2

scorers = {
    'accuracy': make_scorer(accuracy_score),
    'f1': make_scorer(f1_score),
    'recall': make_scorer(recall_score),
    'precision': make_scorer(precision_score)
}

def model_logging(modelname, Iters, random_search, X_valid_tfidf, y_valid):
    ID_lst = [f'{modelname}-{str(j).zfill(4)}' for j in range(1, 1+Iters)]
    val = random_search.cv_results_

    para_df = pd.DataFrame.from_dict(val['params'])
    para_df['ID'] = ID_lst
    para_df.to_csv(os.path.join(result_dir, f'{modelname}_Parameters.csv'))

    result_train_df = pd.DataFrame()
    result_train_df['ID'] = ID_lst
    result_train_df['Accuracy'] = list(val['mean_train_accuracy'])
    result_train_df['F1-Score'] = list(val['mean_train_f1'])
    result_train_df['Recall'] = list(val['mean_train_recall'])
    result_train_df['Precision'] = list(val['mean_train_precision'])
    result_train_df['Time (Sec)'] = list(val['mean_score_time'])
    result_train_df.to_csv(os.path.join(result_dir, f'{modelname}_Train_result.csv'))
    
    result_test_df = pd.DataFrame()
    result_test_df['ID'] = ID_lst
    result_test_df['Accuracy'] = list(val['mean_test_accuracy'])
    result_test_df['F1-Score'] = list(val['mean_test_f1'])
    result_test_df['Recall'] = list(val['mean_test_recall'])
    result_test_df['Precision'] = list(val['mean_test_precision'])
    result_test_df['Time (Sec)'] = list(val['mean_fit_time'])
    result_test_df.to_csv(os.path.join(result_dir, f'{modelname}_Test_result.csv'))
    
    y_pred = random_search.predict(X_valid_tfidf)
    report = classification_report(y_valid, y_pred, digits=5)
    
    with open(os.path.join(result_dir, f'{modelname}_Valid_report.txt'), 'w') as f:
        f.write(report)

RFC_para = {
    'n_estimators': range(100, 551, 50),
    'criterion': ['gini', 'entropy'],
    'max_features': ['sqrt', 'log2'],
    'max_depth': [None] + list(range(1, 100)),
    'min_samples_split': list(np.arange(0.1, 1, 0.1)) + list(range(2,100)),
    'min_samples_leaf': list(np.arange(0.1, 0.5, 0.1)) + list(range(1,100)),
    'random_state': [RANDOM_STATE]
}

random_search = RandomizedSearchCV(
    estimator=RandomForestClassifier(),
    param_distributions=RFC_para,
    n_iter=Iters,
    n_jobs=parallel_workers,
    cv=cross_val_works,
    scoring=scorers,
    refit='accuracy',
    return_train_score=True,
    verbose=verbose
)

random_search.fit(X_train_tfidf, y_train)
model_logging('RFC',Iters,random_search,X_valid_tfidf,y_valid)
