import os
import random
import pandas as pd
import numpy as np 
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
df = pd.read_csv(os.path.join(dataset_dir,'train_news_preprocessed.csv'), low_memory=False, 
                 usecols = ['label','clean_news_tokens'])
# 'headline','news','headline_len','news_len','caps_in_headline','caps_in_news','clean_headline_tokens'

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

print('TDIDF')
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train).copy()
X_valid_tfidf = vectorizer.transform(X_valid).copy()

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

estimator.append(('MNB',MultinomialNB(fit_prior=False, alpha=0.03)))
estimator.append(('RFC',RandomForestClassifier(n_estimators=150, min_samples_split=44, min_samples_leaf=1, max_features='sqrt', max_depth=87, criterion='gini', random_state=RANDOM_STATE)))
estimator.append(('GBC',GradientBoostingClassifier(n_estimators=500, min_samples_split=73, min_samples_leaf=20, max_features='sqrt', max_depth=24, learning_rate=0.09, criterion='friedman_mse',  random_state=RANDOM_STATE)))
estimator.append(('LR',LogisticRegression(solver='saga', penalty='l1', max_iter=1000, C=67.85, random_state=RANDOM_STATE)))
estimator.append(('SVM',SVC(kernel='rbf', gamma=0.76, degree=1, C=6.23, random_state=RANDOM_STATE)))

vot_hard = VotingClassifier(estimators = estimator, voting ='hard', verbose=verbose, weights=[1,1,2,1.5,2])

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

y_pred_valid = best_estimator.predict(X_valid_tfidf)

report = classification_report(y_valid, y_pred_valid, digits=5)
print(report)
with open(os.path.join(result_dir, f'{modelname}_Valid_report.txt'), 'w') as f:
    f.write(report)
