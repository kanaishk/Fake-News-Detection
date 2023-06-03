import os
import random
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.metrics import make_scorer, accuracy_score, f1_score, recall_score, precision_score
from sklearn.metrics import classification_report

print('Start')
cwd = os.getcwd()
dataset_dir = os.path.join(cwd,'Dataset')
result_dir = os.path.join(cwd,'Results')
model_dir = os.path.join(cwd,'Model')
df = pd.read_csv(os.path.join(dataset_dir,'train_news_preprocessed_mc.csv'), low_memory=False, 
                 usecols = ['label','clean_news_tokens'])

RANDOM_STATE = 1973
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
pd.options.display.max_seq_items = 20
pd.options.display.max_rows = 50

print('Tokens')
X = df['clean_news_tokens'].apply(lambda x: ' '.join(eval(x)))  # Concatenate tokens into space-separated strings
y = df['label']  # Target variable (fake or not fake)

print('Splitter')
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

print('TFIDF')
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train).copy()

parallel_workers = 5
cross_val_works = 5
verbose = 4

scorers = {
    'accuracy': make_scorer(accuracy_score),
    'f1': make_scorer(f1_score),
    'recall': make_scorer(recall_score),
    'precision': make_scorer(precision_score)
}

print('Model')

modelname = 'Voting'

estimator = []

estimator.append(('MNB',MultinomialNB(fit_prior=True, alpha=0.02)))
estimator.append(('RFC',RandomForestClassifier(n_estimators=200, min_samples_split=24, min_samples_leaf=6, max_features='sqrt', max_depth=46, criterion='entropy', random_state=RANDOM_STATE)))
estimator.append(('GBC',GradientBoostingClassifier(n_estimators=350, min_samples_split=92, min_samples_leaf=21, max_features='sqrt', max_depth=54, learning_rate=0.08, criterion='friedman_mse',  random_state=RANDOM_STATE)))
estimator.append(('LR',LogisticRegression(solver='liblinear', penalty='l2', max_iter=1000, C=79.35, random_state=RANDOM_STATE)))
estimator.append(('SVM',SVC(kernel='linear', gamma=1.9, degree=1, C=2.69, random_state=RANDOM_STATE)))

vot_hard = VotingClassifier(estimators = estimator, voting ='hard', verbose=verbose, weights=[1.5,1,2,2,2])

cv_results = cross_validate(vot_hard,
                            X_train_tfidf,
                            y_train,
                            n_jobs=parallel_workers,
                            cv=cross_val_works,
                            scoring=scorers,
                            return_train_score=True,
                            return_estimator=True)

# Calculate mean scores from cross-validation results
mean_train_accuracy = np.mean(cv_results['train_accuracy'])
mean_train_f1 = np.mean(cv_results['train_f1'])
mean_train_recall = np.mean(cv_results['train_recall'])
mean_train_precision = np.mean(cv_results['train_precision'])
mean_score_time = np.mean(cv_results['score_time'])

mean_test_accuracy = np.mean(cv_results['test_accuracy'])
mean_test_f1 = np.mean(cv_results['test_f1'])
mean_test_recall = np.mean(cv_results['test_recall'])
mean_test_precision = np.mean(cv_results['test_precision'])
mean_fit_time = np.mean(cv_results['fit_time'])

# Store mean scores in result dataframes
result_train_df = pd.DataFrame()
result_train_df['Accuracy'] = [mean_train_accuracy]
result_train_df['F1-Score'] = [mean_train_f1]
result_train_df['Recall'] = [mean_train_recall]
result_train_df['Precision'] = [mean_train_precision]
result_train_df['Time (Sec)'] = [mean_score_time]
result_train_df.to_csv(os.path.join(result_dir, f'{modelname}_Train_result.csv'))

result_test_df = pd.DataFrame()
result_test_df['Accuracy'] = [mean_test_accuracy]
result_test_df['F1-Score'] = [mean_test_f1]
result_test_df['Recall'] = [mean_test_recall]
result_test_df['Precision'] = [mean_test_precision]
result_test_df['Time (Sec)'] = [mean_fit_time]
result_test_df.to_csv(os.path.join(result_dir, f'{modelname}_Test_result.csv'))

best_estimator = cv_results['estimator'][np.argmax(cv_results['test_accuracy'])]
best_estimator.fit(X_train_tfidf, y_train)

with open(os.path.join(model_dir,'trained_tuned_model_mc.pkl'), 'wb') as f:
    pickle.dump(best_estimator, f)

