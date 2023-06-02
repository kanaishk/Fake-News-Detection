#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys


# In[2]:


get_ipython().system('{sys.executable} -m pip install pandas')
get_ipython().system('{sys.executable} -m pip install numpy')
get_ipython().system('{sys.executable} -m pip install regex')
get_ipython().system('{sys.executable} -m pip install matplotlib')
get_ipython().system('{sys.executable} -m pip install seaborn')


# In[3]:


import os
import pandas as pd
import numpy as np 
from re import search


# In[4]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[5]:


import warnings
warnings.filterwarnings("ignore")


# <h3>Importing the dataset</h3>

# In[6]:


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
# In[7]:


modelname = ['GBC','LR','MNB','RFC','SVM']
modelnamebase = ['GBC','LR','MNB','RFC','SVM']


# In[8]:


test_res = []
train_res = []
valid_res = []


# In[9]:


base_test_res = []
base_train_res = []
base_valid_res = []


# <h3>Result Processing</h3>

# <h4>Base</h4>

# In[10]:


for model in modelnamebase:
    base_train_df = pd.read_csv(os.path.join(result_dir, f'BASE\BASE_{model}_Train_result.csv'))
    base_test_df = pd.read_csv(os.path.join(result_dir, f'BASE\BASE_{model}_Test_result.csv'))
    
    unnamed_columns = [col for col in base_test_df.columns if search(r'^Unnamed', col)]
    base_test_df = base_test_df.drop(unnamed_columns, axis=1)
    base_test_df.drop('Time (Sec)', axis=1, inplace=True)
    
    unnamed_columns = [col for col in base_train_df.columns if search(r'^Unnamed', col)]
    base_train_df = base_train_df.drop(unnamed_columns, axis=1)
    base_train_df.drop('Time (Sec)', axis=1, inplace=True)
    
    base_test_df.drop('ID', axis=1, inplace=True)
    base_train_df.drop('ID', axis=1, inplace=True)
    
    base_final_test_res = base_test_df.to_dict(orient='records')[0]
    base_final_test_res['Model'] = model
    base_test_res.append(base_final_test_res)
    
    base_final_train_res = base_train_df.to_dict(orient='records')[0]
    base_final_train_res['Model'] = model
    base_train_res.append(base_final_train_res)


# In[11]:


for model in modelnamebase :
    file_path = os.path.join(result_dir,f'BASE\BASE_{model}_Valid_report.txt')

    with open(file_path, 'r') as file:
        classification_report_str = file.read()

    lines = classification_report_str.strip().split('\n')

    accuracy_line = lines[-3].split()[1:]
    macro_avg_line = lines[-2].split()[2:-1]

    accuracy = float(accuracy_line[0])
    macro_avg_precision, macro_avg_recall, macro_avg_f1 = map(float, macro_avg_line)
    
    base_valid_res_mdl = dict()
    base_valid_res_mdl['Accuracy'] = accuracy
    base_valid_res_mdl['F1-Score'] = macro_avg_f1
    base_valid_res_mdl['Recall'] = macro_avg_recall
    base_valid_res_mdl['Precision'] = macro_avg_precision
    base_valid_res_mdl['Model']=model
    
    base_valid_res.append(base_valid_res_mdl)


# In[12]:


base_test_res_df = pd.DataFrame(base_test_res)
base_test_res_df['Model'] = base_test_res_df['Model'].astype(str)
base_test_res_df.head(6)


# In[13]:


base_train_res_df = pd.DataFrame(base_train_res)
base_train_res_df['Model'] = base_train_res_df['Model'].astype(str)
base_train_res_df.head(6)


# In[14]:


base_valid_res_df = pd.DataFrame(base_valid_res)
base_valid_res_df['Model'] = base_valid_res_df['Model'].astype(str)
base_valid_res_df.head(6)


# <h4>Hyper Tuned</h4>

# In[15]:


for model in modelname :
    train_df = pd.read_csv(os.path.join(result_dir,f'{model}_Train_result.csv'))
    test_df = pd.read_csv(os.path.join(result_dir,f'{model}_Test_result.csv'))
    parameter_df = pd.read_csv(os.path.join(result_dir,f'{model}_Parameters.csv'))
    
    unnamed_columns = [col for col in test_df.columns if search(r'^Unnamed', col)]
    test_df = test_df.drop(unnamed_columns, axis=1)
    test_columns = list(test_df.columns)
    
    test_top_ten = pd.DataFrame()
    test_top_ten = test_top_ten.append(test_df.nlargest(10, ['Accuracy']), ignore_index = True)
    test_top_ten.columns = test_columns
    test_top_ten.drop('Time (Sec)', axis=1, inplace=True)
    
    print('Test')
    print(test_top_ten)
    ID_list = list(test_top_ten['ID'])
    
    unnamed_columns = [col for col in train_df.columns if search(r'^Unnamed', col)]
    train_df = train_df.drop(unnamed_columns, axis=1)
    train_columns = list(train_df.columns)
    
    train_ten = pd.DataFrame()
    for ID in ID_list :
        train_ten = train_ten.append([train_df[train_df.ID == ID]], ignore_index = True)
    train_ten.columns = train_columns
    train_ten.drop('Time (Sec)', axis=1, inplace=True)
    
    #print('Train')
    #print(train_ten)
    
    unnamed_columns = [col for col in parameter_df.columns if search(r'^Unnamed', col)]
    parameter_df = parameter_df.drop(unnamed_columns, axis=1)
    parameter_columns = list(parameter_df.columns)
    
    parameter_ten = pd.DataFrame()
    for ID in ID_list :
        parameter_ten = parameter_ten.append([parameter_df[parameter_df.ID == ID]], ignore_index = True)
    parameter_ten.columns = parameter_columns
    
    #print(f'Parameters {model}')
    #print(parameter_ten)
    
    final_test_res = dict(test_top_ten.mean())
    final_test_res['Model'] = model
    test_res.append(final_test_res)
    
    final_train_res = dict(train_ten.mean())
    final_train_res['Model'] = model
    train_res.append(final_train_res)


# In[16]:


for model in modelname :
    file_path = os.path.join(result_dir,f'{model}_Valid_report.txt')

    with open(file_path, 'r') as file:
        classification_report_str = file.read()

    lines = classification_report_str.strip().split('\n')

    accuracy_line = lines[-3].split()[1:]
    macro_avg_line = lines[-2].split()[2:-1]

    accuracy = float(accuracy_line[0])
    macro_avg_precision, macro_avg_recall, macro_avg_f1 = map(float, macro_avg_line)
    
    valid_res_mdl = dict()
    valid_res_mdl['Accuracy'] = accuracy
    valid_res_mdl['F1-Score'] = macro_avg_f1
    valid_res_mdl['Recall'] = macro_avg_recall
    valid_res_mdl['Precision'] = macro_avg_precision
    valid_res_mdl['Model']=model
    
    valid_res.append(valid_res_mdl)


# In[17]:


test_res_df = pd.DataFrame(test_res)
test_res_df['Model'] = test_res_df['Model'].astype(str)
test_res_df.head(6)


# In[18]:


train_res_df = pd.DataFrame(train_res)
train_res_df['Model'] = train_res_df['Model'].astype(str)
train_res_df.head(6)


# In[19]:


valid_res_df = pd.DataFrame(valid_res)
valid_res_df['Model'] = valid_res_df['Model'].astype(str)
valid_res_df.head(6)


# <h3>Graphing</h3>

# In[20]:


graph_dir = os.path.join(cwd,'Graphs')


# <h3>Base</h3>

# In[21]:


plt.figure(figsize=(10, 6))
sns.barplot(x='Model', y='Accuracy', data=base_train_res_df)
title='Training Accuracy of Base Models'
plt.title(title)
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.savefig(os.path.join(graph_dir,title+'.png'), bbox_inches='tight')
plt.show()
plt.close()


# In[22]:


plt.figure(figsize=(10, 6))
sns.barplot(x='Model', y='Accuracy', data=base_test_res_df)
title = 'Base Test Accuracy of Base Models'
plt.title(title)
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.savefig(os.path.join(graph_dir,title+'.png'), bbox_inches='tight')
plt.show()
plt.close()


# In[23]:


plt.figure(figsize=(10, 6))
sns.barplot(x='Model', y='Accuracy', data=base_valid_res_df)
title='Base Validation Accuracy of Base Models'
plt.title(title)
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.savefig(os.path.join(graph_dir,title+'.png'), bbox_inches='tight')
plt.show()
plt.close()


# In[24]:


merged_base_df = pd.merge(base_train_res_df, base_test_res_df, on='Model', suffixes=('_train', '_test'))
merged_base_df = pd.merge(merged_base_df, base_valid_res_df, on='Model')

plt.figure(figsize=(10, 6))

plt.plot(merged_base_df['Model'], merged_base_df['Accuracy_train'], marker='o', label='Training Accuracy')
for x, y in zip(merged_base_df['Model'], merged_base_df['Accuracy_train']):
    plt.text(x, y, f'{y:.4f}', ha='center', va='bottom')

plt.plot(merged_base_df['Model'], merged_base_df['Accuracy_test'], marker='o', label='Testing Accuracy')
for x, y in zip(merged_base_df['Model'], merged_base_df['Accuracy_test']):
    plt.text(x, y-0.011, f'{y:.4f}', ha='center', va='bottom')

plt.plot(merged_base_df['Model'], merged_base_df['Accuracy'], marker='o', label='Validation Accuracy')
for x, y in zip(merged_base_df['Model'], merged_base_df['Accuracy']):
    plt.text(x, y, f'{y:.4f}', ha='center', va='bottom')

title = 'Base Model Accuracy Comparison'
plt.title(title)
plt.xlabel('Model')
plt.ylabel('Accuracy')

plt.legend()
plt.savefig(os.path.join(graph_dir,title+'.png'), bbox_inches='tight')
plt.show()
plt.close()


# In[25]:


# Create a figure with 2x2 subplots
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))

# Accuracy subgraph
axes[0, 0].plot(merged_base_df['Model'], merged_base_df['Accuracy_train'], marker='o', label='Training Accuracy')
axes[0, 0].plot(merged_base_df['Model'], merged_base_df['Accuracy_test'], marker='o', label='Testing Accuracy')
axes[0, 0].plot(merged_base_df['Model'], merged_base_df['Accuracy'], marker='o', label='Validation Accuracy')
axes[0, 0].set_title('Accuracy')
axes[0, 0].set_xlabel('Model')
axes[0, 0].set_ylabel('Accuracy')
axes[0, 0].legend()
for x, y in zip(merged_base_df['Model'], merged_base_df['Accuracy']):
    axes[0, 0].text(x, y, f'{y:.4f}', ha='center', va='bottom')

# Precision subgraph
axes[0, 1].plot(merged_base_df['Model'], merged_base_df['Precision_train'], marker='o', label='Training Precision')
axes[0, 1].plot(merged_base_df['Model'], merged_base_df['Precision_test'], marker='o', label='Testing Precision')
axes[0, 1].plot(merged_base_df['Model'], merged_base_df['Precision'], marker='o', label='Validation Precision')
axes[0, 1].set_title('Precision')
axes[0, 1].set_xlabel('Model')
axes[0, 1].set_ylabel('Precision')
axes[0, 1].legend()
for x, y in zip(merged_base_df['Model'], merged_base_df['Precision']):
    axes[0, 1].text(x, y, f'{y:.4f}', ha='center', va='bottom')

# Recall subgraph
axes[1, 0].plot(merged_base_df['Model'], merged_base_df['Recall_train'], marker='o', label='Training Recall')
axes[1, 0].plot(merged_base_df['Model'], merged_base_df['Recall_test'], marker='o', label='Testing Recall')
axes[1, 0].plot(merged_base_df['Model'], merged_base_df['Recall'], marker='o', label='Validation Recall')
axes[1, 0].set_title('Recall')
axes[1, 0].set_xlabel('Model')
axes[1, 0].set_ylabel('Recall')
axes[1, 0].legend()
for x, y in zip(merged_base_df['Model'], merged_base_df['Recall']):
    axes[1, 0].text(x, y, f'{y:.4f}', ha='center', va='bottom')

# F1 Score subgraph
axes[1, 1].plot(merged_base_df['Model'], merged_base_df['F1-Score_train'], marker='o', label='Training F1 Score')
axes[1, 1].plot(merged_base_df['Model'], merged_base_df['F1-Score_test'], marker='o', label='Testing F1 Score')
axes[1, 1].plot(merged_base_df['Model'], merged_base_df['F1-Score'], marker='o', label='Validation F1 Score')
axes[1, 1].set_title('F1 Score')
axes[1, 1].set_xlabel('Model')
axes[1, 1].set_ylabel('F1 Score')
axes[1, 1].legend()
for x, y in zip(merged_base_df['Model'], merged_base_df['F1-Score']):
    axes[1, 1].text(x, y, f'{y:.4f}', ha='center', va='bottom')

# Adjust the spacing between subplots
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# Save and display the graph
title = 'Base Model Performance Comparison'
plt.suptitle(title, fontsize=16)
plt.savefig(os.path.join(graph_dir, title+'.png'), bbox_inches='tight')
plt.show()
plt.close()


# <h3>Hyper Tuned</h3>

# In[26]:


plt.figure(figsize=(10, 6))
sns.barplot(x='Model', y='Accuracy', data=train_res_df)
title='Training Accuracy of Hyperparameter Tuned Model'
plt.title(title)
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.savefig(os.path.join(graph_dir,title+'.png'), bbox_inches='tight')
plt.show()
plt.close()


# In[27]:


plt.figure(figsize=(10, 6))
sns.barplot(x='Model', y='Accuracy', data=test_res_df)
title = 'Test Accuracy of Hyperparameter Tuned Model'
plt.title(title)
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.savefig(os.path.join(graph_dir,title+'.png'), bbox_inches='tight')
plt.show()
plt.close()


# In[28]:


plt.figure(figsize=(10, 6))
sns.barplot(x='Model', y='Accuracy', data=valid_res_df)
title='Validation Accuracy of Hyperparameter Tuned Model'
plt.title(title)
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.savefig(os.path.join(graph_dir,title+'.png'), bbox_inches='tight')
plt.show()
plt.close()


# In[29]:


merged_tuned_df = pd.merge(train_res_df, test_res_df, on='Model', suffixes=('_train', '_test'))
merged_tuned_df = pd.merge(merged_tuned_df, valid_res_df, on='Model')

plt.figure(figsize=(10, 6))

plt.plot(merged_tuned_df['Model'], merged_tuned_df['Accuracy_train'], marker='o', label='Training Accuracy')
for x, y in zip(merged_tuned_df['Model'], merged_tuned_df['Accuracy_train']):
    plt.text(x, y, f'{y:.4f}', ha='center', va='bottom')

plt.plot(merged_tuned_df['Model'], merged_tuned_df['Accuracy_test'], marker='o', label='Testing Accuracy')
for x, y in zip(merged_tuned_df['Model'], merged_tuned_df['Accuracy_test']):
    plt.text(x, y, f'{y:.4f}', ha='center', va='bottom')

plt.plot(merged_tuned_df['Model'], merged_tuned_df['Accuracy'], marker='o', label='Validation Accuracy')
for x, y in zip(merged_tuned_df['Model'], merged_tuned_df['Accuracy']):
    plt.text(x, y-0.005, f'{y:.4f}', ha='center', va='bottom')

title='Tuned Model Accuracy Comparison'
plt.title(title)
plt.xlabel('Model')
plt.ylabel('Accuracy')

plt.legend()
plt.savefig(os.path.join(graph_dir,title+'.png'), bbox_inches='tight')
plt.show()
plt.close()


# In[30]:


# Create a figure with 2x2 subplots
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))

# Accuracy subgraph
axes[0, 0].plot(merged_tuned_df['Model'], merged_tuned_df['Accuracy_train'], marker='o', label='Training Accuracy')
axes[0, 0].plot(merged_tuned_df['Model'], merged_tuned_df['Accuracy_test'], marker='o', label='Testing Accuracy')
axes[0, 0].plot(merged_tuned_df['Model'], merged_tuned_df['Accuracy'], marker='o', label='Validation Accuracy')
axes[0, 0].set_title('Accuracy')
axes[0, 0].set_xlabel('Model')
axes[0, 0].set_ylabel('Accuracy')
axes[0, 0].legend()
for x, y in zip(merged_tuned_df['Model'], merged_tuned_df['Accuracy']):
    axes[0, 0].text(x, y, f'{y:.4f}', ha='center', va='bottom')

# Precision subgraph
axes[0, 1].plot(merged_tuned_df['Model'], merged_tuned_df['Precision_train'], marker='o', label='Training Precision')
axes[0, 1].plot(merged_tuned_df['Model'], merged_tuned_df['Precision_test'], marker='o', label='Testing Precision')
axes[0, 1].plot(merged_tuned_df['Model'], merged_tuned_df['Precision'], marker='o', label='Validation Precision')
axes[0, 1].set_title('Precision')
axes[0, 1].set_xlabel('Model')
axes[0, 1].set_ylabel('Precision')
axes[0, 1].legend()
for x, y in zip(merged_tuned_df['Model'], merged_tuned_df['Precision']):
    axes[0, 1].text(x, y, f'{y:.4f}', ha='center', va='bottom')

# Recall subgraph
axes[1, 0].plot(merged_tuned_df['Model'], merged_tuned_df['Recall_train'], marker='o', label='Training Recall')
axes[1, 0].plot(merged_tuned_df['Model'], merged_tuned_df['Recall_test'], marker='o', label='Testing Recall')
axes[1, 0].plot(merged_tuned_df['Model'], merged_tuned_df['Recall'], marker='o', label='Validation Recall')
axes[1, 0].set_title('Recall')
axes[1, 0].set_xlabel('Model')
axes[1, 0].set_ylabel('Recall')
axes[1, 0].legend()
for x, y in zip(merged_tuned_df['Model'], merged_tuned_df['Recall']):
    axes[1, 0].text(x, y, f'{y:.4f}', ha='center', va='bottom')

# F1 Score subgraph
axes[1, 1].plot(merged_tuned_df['Model'], merged_tuned_df['F1-Score_train'], marker='o', label='Training F1 Score')
axes[1, 1].plot(merged_tuned_df['Model'], merged_tuned_df['F1-Score_test'], marker='o', label='Testing F1 Score')
axes[1, 1].plot(merged_tuned_df['Model'], merged_tuned_df['F1-Score'], marker='o', label='Validation F1 Score')
axes[1, 1].set_title('F1 Score')
axes[1, 1].set_xlabel('Model')
axes[1, 1].set_ylabel('F1 Score')
axes[1, 1].legend()
for x, y in zip(merged_tuned_df['Model'], merged_tuned_df['F1-Score']):
    axes[1, 1].text(x, y, f'{y:.4f}', ha='center', va='bottom')

# Adjust the spacing between subplots
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# Save and display the graph
title = 'Tuned Model Performance Comparison'
plt.suptitle(title, fontsize=16)
plt.savefig(os.path.join(graph_dir, title+'.png'), bbox_inches='tight')
plt.show()
plt.close()


# <h3>Others</h3>

# In[31]:


# Calculate the difference in accuracies between base and tuned models
merged_diff_df = pd.DataFrame()
merged_diff_df['Model'] = merged_base_df['Model']
merged_diff_df['Diff_Accuracy_train'] = merged_tuned_df['Accuracy_train'] - merged_base_df['Accuracy_train']
merged_diff_df['Diff_Accuracy_test'] = merged_tuned_df['Accuracy_test'] - merged_base_df['Accuracy_test']
merged_diff_df['Diff_Accuracy'] = merged_tuned_df['Accuracy'] - merged_base_df['Accuracy']

# Plot the difference in accuracies
plt.figure(figsize=(10, 6))

plt.plot(merged_diff_df['Model'], merged_diff_df['Diff_Accuracy_train'], marker='o', label='Training Accuracy')
for x, y in zip(merged_diff_df['Model'], merged_diff_df['Diff_Accuracy_train']):
    plt.text(x, y+0.002, f'{y:.4f}', ha='center', va='bottom')

plt.plot(merged_diff_df['Model'], merged_diff_df['Diff_Accuracy_test'], marker='o', label='Testing Accuracy')
for x, y in zip(merged_diff_df['Model'], merged_diff_df['Diff_Accuracy_test']):
    plt.text(x, y-0.005, f'{y:.4f}', ha='center', va='bottom')

plt.plot(merged_diff_df['Model'], merged_diff_df['Diff_Accuracy'], marker='o', label='Validation Accuracy')
for x, y in zip(merged_diff_df['Model'], merged_diff_df['Diff_Accuracy']):
    plt.text(x, y-0.012, f'{y:.4f}', ha='center', va='bottom')

title = 'Difference in Accuracy-Base vs Tuned Models'
plt.title(title)
plt.xlabel('Model')
plt.ylabel('Accuracy Difference')

plt.legend()
plt.savefig(os.path.join(graph_dir, title+'.png'), bbox_inches='tight')
plt.show()
plt.close()


# In[ ]:




