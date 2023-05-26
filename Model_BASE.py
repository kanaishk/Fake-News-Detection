import os
import random
import pandas as pd
import numpy as np 
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import make_scorer, accuracy_score, f1_score, recall_score, precision_score
from sklearn.metrics import classification_report

cwd = os.getcwd()
dataset_dir = os.path.join(cwd,'Dataset')
result_dir = os.path.join(cwd,'Results/Count')
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

transformer = TfidfTransformer(smooth_idf=False)
vectorizer = CountVectorizer(ngram_range=(1, 2))
X_train_count = vectorizer.fit_transform(X_train)
X_valid_count = vectorizer.transform(X_valid)
#X_train_count = transformer.fit_transform(X_train_count)
#X_valid_count = transformer.fit_transform(X_valid_count)

Iters = 1
parallel_workers = 5
cross_val_works = 5
verbose = 2

scorers = {
    'accuracy': make_scorer(accuracy_score),
    'f1': make_scorer(f1_score),
    'recall': make_scorer(recall_score),
    'precision': make_scorer(precision_score)
}

def model_logging(modelname, Iters, random_search, X_valid_tfidf, y_valid):
    ID_lst = f'{modelname}-BASE'
    val = random_search.cv_results_

    para_df = pd.DataFrame.from_dict(val['params'])
    para_df['ID'] = ID_lst
    para_df.to_csv(os.path.join(result_dir, f'BASE/BASE_{modelname}_Parameters.csv'))

    result_train_df = pd.DataFrame()
    result_train_df['ID'] = ID_lst
    result_train_df['Accuracy'] = list(val['mean_train_accuracy'])
    result_train_df['F1-Score'] = list(val['mean_train_f1'])
    result_train_df['Recall'] = list(val['mean_train_recall'])
    result_train_df['Precision'] = list(val['mean_train_precision'])
    result_train_df['Time (Sec)'] = list(val['mean_score_time'])
    result_train_df.to_csv(os.path.join(result_dir, f'BASE/BASE_{modelname}_Train_result.csv'))
    
    result_test_df = pd.DataFrame()
    result_test_df['ID'] = ID_lst
    result_test_df['Accuracy'] = list(val['mean_test_accuracy'])
    result_test_df['F1-Score'] = list(val['mean_test_f1'])
    result_test_df['Recall'] = list(val['mean_test_recall'])
    result_test_df['Precision'] = list(val['mean_test_precision'])
    result_test_df['Time (Sec)'] = list(val['mean_fit_time'])
    result_test_df.to_csv(os.path.join(result_dir, f'BASE/BASE_{modelname}_Test_result.csv'))
    
    y_pred = random_search.predict(X_valid_tfidf)
    report = classification_report(y_valid, y_pred, digits=5)
    
    with open(os.path.join(result_dir, f'BASE/BASE_{modelname}_Valid_report.txt'), 'w') as f:
        f.write(report)

RFC_para = {
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

random_search.fit(X_train_count, y_train)
model_logging('RFC',Iters,random_search,X_valid_count,y_valid)

random_search = RandomizedSearchCV(
    estimator=KNeighborsClassifier(),
    param_distributions={},
    n_iter=Iters,
    n_jobs=parallel_workers,
    cv=cross_val_works,
    scoring=scorers,
    refit='accuracy',
    return_train_score=True,
    verbose=verbose
)

random_search.fit(X_train_count, y_train)
model_logging('KNN',Iters,random_search,X_valid_count,y_valid)

LR_para = {
    'random_state': [RANDOM_STATE]
}

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

random_search.fit(X_train_count, y_train)
model_logging('LR',Iters,random_search,X_valid_count,y_valid)

random_search = RandomizedSearchCV(
    estimator=MultinomialNB(),
    param_distributions={},
    n_iter=Iters,
    n_jobs=parallel_workers,
    cv=cross_val_works,
    scoring=scorers,
    refit='accuracy',
    return_train_score=True,
    verbose=verbose
)

random_search.fit(X_train_count, y_train)
model_logging('MNB',Iters,random_search,X_valid_count,y_valid)

GBC_para = {
    'random_state': [RANDOM_STATE]
}

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

random_search.fit(X_train_count, y_train)
model_logging('GBC',Iters,random_search,X_valid_count,y_valid)

SVM_para = {
    'random_state': [RANDOM_STATE]
}

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

random_search.fit(X_train_count, y_train)
model_logging('SVM',Iters,random_search,X_valid_count,y_valid)
