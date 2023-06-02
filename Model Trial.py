#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import sys


# In[ ]:


get_ipython().system('{sys.executable} -m pip install pandas')
get_ipython().system('{sys.executable} -m pip install numpy')
get_ipython().system('{sys.executable} -m pip install scikit-learn')


# In[ ]:


import os
import random
import pandas as pd
import numpy as np 


# In[ ]:


from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import make_scorer, accuracy_score, f1_score, recall_score, precision_score
from sklearn.metrics import classification_report


# <h3>Importing the dataset</h3>

# In[ ]:


cwd = os.getcwd()
dataset_dir = os.path.join(cwd,'Dataset')
result_dir = os.path.join(cwd,'Results')

Data- Description: There are 11 columns in the dataset provided to you. The description of each of the column is given below: 
“id”: Unique id of each news article 
“headline”: It is the title of the news. 
“written_by”: It represents the author of the news article
“news”: It contains the full text of the news article 
“label”: It tells whether the news is fake (1) or not fake (0).
"headline_len": length of headline
"news_len": length of news article
"caps_in_headline": number of capital letter in headline
"norm_caps_in_headline": percentage of capital in headline
"caps_in_news": number of capital letter in news article
"norm_caps_in_news": percentage of capital in news article
"news_tokens": tokenized news article
"news_url": urls in news article
"clean_news": cleaned up news article
"headline_url": urls in headlines
"clean_headline": cleaned up news headline
"twitter_handles": twitter handles in news article
"clean_news_tokens": tokens of cleaned news article
"clean_headline_tokens": tokens of cleaned headline
# In[ ]:


df = pd.read_csv(os.path.join(dataset_dir,'train_news_preprocessed.csv'), low_memory=False, 
                 usecols = ['label','clean_news_tokens','clean_headline_tokens'])
# 'headline','news','headline_len','news_len','caps_in_headline','caps_in_news',


# <h3>Setting environment variables</h3>

# In[ ]:


RANDOM_STATE = 1973
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
pd.options.display.max_seq_items = 20
pd.options.display.max_rows = 50


# <h3>Starting model implementation</h3>

# In[ ]:


X = df['clean_news_tokens'].apply(lambda x: ' '.join(eval(x)))  # Concatenate tokens into space-separated strings
y = df['label']  # Target variable (fake or not fake)


# In[ ]:


X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)


# In[ ]:


transformer = TfidfTransformer(smooth_idf=False)
vectorizer = CountVectorizer(ngram_range=(1, 2))
X_train_count = vectorizer.fit_transform(X_train)
X_valid_count = vectorizer.transform(X_valid)


# In[ ]:


X_train_count = transformer.fit_transform(X_train_count)
X_valid_count = transformer.fit_transform(X_valid_count)


# In[ ]:


vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_valid_tfidf = vectorizer.transform(X_valid)


# In[ ]:


Iters = 100
parallel_workers = 9
cross_val_works = 5
verbose = 2


# In[ ]:


scorers = {
    'accuracy': make_scorer(accuracy_score),
    'f1': make_scorer(f1_score),
    'recall': make_scorer(recall_score),
    'precision': make_scorer(precision_score)
}


# In[ ]:


def model_logging(modelname, Iters, random_search):
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


# In[ ]:


def model_logging_valid(modelname, random_search, X_valid_tfidf, y_valid):
    y_pred = random_search.predict(X_valid_tfidf)
    report = classification_report(y_valid, y_pred, digits=5)
    
    with open(os.path.join(result_dir, f'{modelname}_Valid_report.txt'), 'w') as f:
        f.write(report)


# <h3>Naive Bayes</h3>

# In[ ]:


MNB_para = {
    'alpha': np.arange(1e-2, 2, 1e-2),
    'fit_prior': [True, False]
    #'max_iter': [1000],
}


# In[ ]:


random_search = RandomizedSearchCV(
    estimator=MultinomialNB(),
    param_distributions=MNB_para,
    n_iter=Iters,
    n_jobs=parallel_workers,
    cv=cross_val_works,
    scoring=scorers,
    refit='accuracy',
    return_train_score=True,
    verbose=verbose
)


# In[ ]:


random_search.fit(X_train_tfidf, y_train)


# In[ ]:


model_logging('MNB',Iters,random_search)
model_logging_valid('MNB',random_search,X_valid_tfidf,y_valid)


# <h3>K Nearest Neighbours</h3>

# In[ ]:


KNN_para = {
    'n_neighbors': range(2, 5),
    'weights': ['uniform', 'distance'],
    'p': [1, 2],
    'leaf_size': range(10, 100, 10)
}


# In[ ]:


random_search = RandomizedSearchCV(
    estimator=KNeighborsClassifier(),
    param_distributions=KNN_para,
    n_iter=Iters,
    n_jobs=parallel_workers,
    cv=cross_val_works,
    scoring=scorers,
    refit='accuracy',
    return_train_score=True,
    verbose=verbose
)


# In[ ]:


random_search.fit(X_train_tfidf, y_train)


# In[ ]:


model_logging('KNN',Iters,random_search)
model_logging_valid('KNN',random_search,X_valid_tfidf,y_valid)


# <h3>Logistic Regression</h3>

# In[ ]:


LR_para = {
    'C': np.arange(1e-2, 1e2, 1e-2),
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear', 'saga'],
    'max_iter': [1000],
    'random_state': [RANDOM_STATE]
}


# In[ ]:


random_search = RandomizedSearchCV(
    estimator=LogisticRegression(),
    param_distributions=LR_para,
    n_iter=Iters,
    n_jobs=parallel_workers,
    cv=cross_val_works,
    scoring=scorers,
    refit='accuracy',
    return_train_score=True,
    verbose=verbose
)


# In[ ]:


random_search.fit(X_train_tfidf, y_train)


# In[ ]:


model_logging('LR',Iters,random_search)
model_logging_valid('LR',random_search,X_valid_tfidf,y_valid)


# <h3>Gradient Boosting</h3>

# In[ ]:


GBC_para = {
    'n_estimators': range(100, 551, 50),
    'criterion': ['friedman_mse'],
    'learning_rate': np.arange(1e-2, 0.1, 1e-2),
    'max_features': ['sqrt', 'log2'],
    'max_depth': [None] + list(range(1, 100)),
    'min_samples_split': list(np.arange(0.1, 1, 0.1)) + list(range(2,100)),
    'min_samples_leaf': list(np.arange(0.1, 0.5, 0.1)) + list(range(1,100)),
    #'max_iter': [1000],
    'random_state': [RANDOM_STATE]
}


# In[ ]:


random_search = RandomizedSearchCV(
    estimator=GradientBoostingClassifier(),
    param_distributions=GBC_para,
    n_iter=Iters,
    n_jobs=parallel_workers,
    cv=cross_val_works,
    scoring=scorers,
    refit='accuracy',
    return_train_score=True,
    verbose=verbose
)


# In[ ]:


random_search.fit(X_train_tfidf, y_train)


# In[ ]:


model_logging('GBC',Iters,random_search)
model_logging_valid('GBC',random_search,X_valid_tfidf,y_valid)


# <h3>Random Forest</h3>

# In[ ]:


RFC_para = {
    'n_estimators': range(100, 551, 50),
    'criterion': ['gini', 'entropy'],
    'max_features': ['sqrt', 'log2'],
    'max_depth': [None] + list(range(1, 100)),
    'min_samples_split': list(np.arange(0.1, 1, 0.1)) + list(range(2,100)),
    'min_samples_leaf': list(np.arange(0.1, 0.5, 0.1)) + list(range(1,100)),
    'random_state': [RANDOM_STATE]
}


# In[ ]:


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


# In[ ]:


random_search.fit(X_train_tfidf, y_train)


# In[ ]:


model_logging('RFC',Iters,random_search)
model_logging_valid('RFC',random_search,X_valid_tfidf,y_valid)


# <h3>SVM</h3>

# In[ ]:


SVM_para = {
    'kernel': ['linear', 'poly', 'rbf'],
    'degree': range(1, 6),
    'C': np.arange(1e-2, 10, 1e-2),
    'gamma': ['scale', 'auto'] + list(np.arange(1e-2, 10, 1e-2)),
    'random_state': [RANDOM_STATE]
}


# In[ ]:


random_search = RandomizedSearchCV(
    estimator=SVC(),
    param_distributions=SVM_para,
    n_iter=Iters,
    n_jobs=parallel_workers,
    cv=cross_val_works,
    scoring=scorers,
    refit='accuracy',
    return_train_score=True,
    verbose=verbose
)


# In[ ]:


random_search.fit(X_train_tfidf, y_train)


# In[ ]:


model_logging('SVM',Iters,random_search)
model_logging_valid('SVM',random_search,X_valid_tfidf,y_valid)

