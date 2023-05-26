#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys


# In[2]:


get_ipython().system('{sys.executable} -m pip install numpy')
get_ipython().system('{sys.executable} -m pip install pandas')
get_ipython().system('{sys.executable} -m pip install matplotlib')
get_ipython().system('{sys.executable} -m pip install regex')


# In[3]:


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from re import search


# Preliminary Analysis

# Importing the dataset

# In[4]:


cwd = os.getcwd()
dataset_dir = os.path.join(cwd,'Dataset')
df = pd.read_csv(os.path.join(dataset_dir,'train_news.csv'))

Problem Description
The authenticity of Information has become a longstanding issue affecting businesses and society, both for printed and digital media. On social networks, the reach and effects of information spread occur at such a fast pace and so amplified that distorted, inaccurate, or false information acquires a tremendous potential to cause real-world impacts, within minutes, ffor millions of users. Recently, several public concerns about this problem and some approaches to mitigate the problem were expressed.

Data- Description:
There are 6 columns in the dataset provided to you. The description of each of the column is given below: 
“Unnamed:0”: It is a serial number 
“id”: Unique id of each news article 
“headline”: It is the title of the news. 
“written_by”: It represents the author of the news article 
“news”: It contains the full text of the news article 
“label”: It tells whether the news is fake (1) or not fake (0).
# In[5]:


print("Dataset shape:", df.shape)


# In[6]:


df.head()


# In[7]:


unnamed_columns = [col for col in df.columns if search(r'^Unnamed', col)]

df = df.drop(unnamed_columns, axis=1)

df.head()


# In[8]:


print("Dataset shape:", df.shape)


# Cleaning the data

# Checking for missing data

# In[9]:


df.info()


# In[10]:


print('Dataset null values:\n',df.isna().sum())


# In[11]:


def show_tf_distribution(df, column) :
    null = df[df[column].isna()]
    total = len(null)
    notfake_cnt = list(null[null['label']==0].shape)[0]
    fake_cnt = list(null[null['label']==1].shape)[0]
    
    print('Null Values distribution for \''+column+'\' on basis for realness')
    print('Total:\t',total)
    print('Real %:\t',format(((notfake_cnt/total)*100),'.2f'))
    print('Fake %:\t',format(((fake_cnt/total)*100),'.2f'))


# In[12]:


show_tf_distribution(df, 'news')


# In[13]:


show_tf_distribution(df, 'headline')


# In[14]:


show_tf_distribution(df, 'written_by')


# As every instance of missing value almost always indicates a fake news article, missing value/information can be an identifying factor. Hence not dropping rows with null values.

# Checking for placeholder values and duplicates

# In[15]:


headline_value_counts = df.headline.value_counts()
headline_value_counts[headline_value_counts > 1]


# In[16]:


duplicate_headline_list = set(headline_value_counts[headline_value_counts > 1].keys())
df_dup_headline = df[df.headline.isin(duplicate_headline_list)]
df_dup_headline


# In[17]:


df_dup_headline[df_dup_headline.duplicated()]


# There are no directly duplicated rows

# In[18]:


df_dup_headline[df_dup_headline.duplicated(['headline', 'news'])]


# There are 70 rows with both headlines and news duplicated. These needs to be removed.

# In[19]:


df_dup_headline[df_dup_headline.news == ' ']


# We can leave duplicate headlines as that is a common part of news when it undergoes revision but those instances where both headline and news articles are same needs to be dropped. Rows without news will also be removed.

# Checking news for whitespaces.

# In[20]:


df[df.news == ' ']


# Replacing white spaces with null.

# In[21]:


df = df.replace(r'^\s*$', np.nan, regex=True)


# In[22]:


df.isna().sum()


# This shows that there is increase in null values in column news as we replace articles that only had null values.

# In[23]:


show_tf_distribution(df, 'news')


# The distribution shows that all null news values still point to fake news. As these values are small compared to total dataset size, dropping them might be preferable.

# Removing Duplicated Data

# In[24]:


len(df)


# Dropping rows with no news article or headlines

# In[25]:


df_clean = df.dropna(subset=['headline','news'])


# In[26]:


len(df_clean)


# Dropping rows with same headline and news articles

# In[27]:


df_clean = df_clean.drop_duplicates(['headline', 'news'], ignore_index=True)


# In[28]:


len(df_clean)


# Dropping rows with same news articles

# In[29]:


df_clean = df_clean.drop_duplicates(['news'], ignore_index=True)


# In[30]:


df_clean.info()


# In[31]:


df_clean.isna().sum()


# In[32]:


len(df) - len(df_clean)


# In[33]:


df_clean.label.value_counts()


# Summary

# 935 rows of data removed on basis of not having information in news column, headline column and or having duplicate values.

# Exploring the dataset

# Helper Function

# In[34]:


graph_dir = os.path.join(cwd,'Graphs')


# In[35]:


def show_hist_for_col(df, column, title):
    """
    Display a histogram for a column in a dataframe, splitting the data by label.
    """
    plt.figure(figsize=(12,8))
    df[df.label == 0][column].hist(label='True')
    df[df.label == 1][column].hist(alpha=0.4, label='Fake')
    plt.title(title)
    plt.legend()
    plt.savefig(os.path.join(graph_dir,title+'.png'), bbox_inches='tight')
    plt.show()
    plt.close()


# Distribution of Labels

# In[36]:


plt.figure(figsize=(12,8))
df_clean.label.hist()
title = 'Fake or True News'
plt.title(title)
plt.savefig(os.path.join(graph_dir,title+'.png'), bbox_inches='tight')
plt.show()
plt.close()


# After cleaning the data there is an imbalance present but hopefully not enough to have effect on the models

# Length of Headlines

# In[37]:


df_clean['headline_len'] = df_clean.headline.str.len()


# In[38]:


show_hist_for_col(df_clean, 'headline_len', 'Number of Characters in Headline')


# In[39]:


df_clean[df_clean.label == 0].headline_len.describe()


# In[40]:


df_clean[df_clean.label == 1].headline_len.describe()


# Fake news stories have a wider range in the length of title than True news stories. The quartile differences between fake news articles is much larger than in case of true news article.

# Length of News article

# In[41]:


df_clean['news_len'] = df_clean.news.str.len()


# In[42]:


show_hist_for_col(df_clean, 'news_len', 'Length of Text for News')


# In[43]:


df_clean[df_clean.label == 0].news_len.describe()


# In[44]:


df_clean[df_clean.label == 1].news_len.describe()


# In[45]:


df_long = df_clean[df_clean.news_len > 10000]


# In[46]:


df_long


# In[47]:


df_long.label.value_counts()


# ok there does not seem to be bias in long stories towards either type of articles so they will be kept. also fake news article have shown much higher max story size and significantly lower quartile scores across the board.

# Capital letters in headline.

# In[48]:


df_clean['caps_in_headline'] = df_clean['headline'].apply(lambda headline: sum(1 for char in headline if char.isupper()))


# In[49]:


df_clean


# In[50]:


show_hist_for_col(df_clean, 'caps_in_headline', 'Number of Capitals in Headline')


# In[51]:


df_clean[df_clean.label == 0].caps_in_headline.describe()


# In[52]:


df_clean[df_clean.label == 1].caps_in_headline.describe()


# There is much more deviation in number of Capital letter in headline in fake news articles. also fake news articles have much more amount of Capital letter in headline.

# In[53]:


df_clean['norm_caps_in_headline'] = df_clean['caps_in_headline'] / df_clean['headline_len']


# In[54]:


show_hist_for_col(df_clean, 'norm_caps_in_headline', 'Percentage of Capitals in Headline')


# In[55]:


df_clean[df_clean.label == 0].norm_caps_in_headline.describe()


# In[56]:


df_clean[df_clean.label == 1].norm_caps_in_headline.describe()


# percentage of capitals is not good enough as the values is quite close between fake and true news headlines.

# Capital in news article

# In[57]:


df_clean['caps_in_news'] = df_clean['news'].apply(lambda news: sum(1 for char in news if char.isupper()))


# In[58]:


show_hist_for_col(df_clean, 'caps_in_news', 'Number of Capitals in News')


# In[59]:


df_clean[df_clean.label == 0].caps_in_news.describe()


# In[60]:


df_clean[df_clean.label == 1].caps_in_news.describe()


# In[61]:


df_clean['norm_caps_in_news'] = df_clean['caps_in_news'] / df_clean['news_len']


# In[62]:


show_hist_for_col(df_clean, 'norm_caps_in_news', 'Percentage of Capitals in News')


# In[63]:


df_clean[df_clean.label == 0].norm_caps_in_news.describe()


# In[64]:


df_clean[df_clean.label == 1].norm_caps_in_news.describe()


# Once again there is a lot of overlap in number of capitals in news article between fake and real news. only in third quartile do fake news show much more amount of capitals in fake news articles.

# In[65]:


def check_string_for(substring, fullstring):
    """Check if the substring is in the fullstring"""
    if search(substring, fullstring):
        return True
    else:
        return False


# Via and Image Via in article

# In[66]:


df_via = df_clean[df_clean.news.apply(lambda news_text: check_string_for(' via', news_text))]


# In[67]:


df_via


# In[68]:


df_via['label'].value_counts()


# Via is much more indicative of fake news article compared to true news article.

# In[69]:


df_image_via = df_clean[df_clean.news.apply(lambda news_text: check_string_for('image via', news_text))]


# In[70]:


df_image_via


# In[71]:


df_image_via['label'].value_counts()


# With all the posts with image via being Fake, it's highly indicative of that label, but this may be particular to this dataset and may not generalize.

# Said in news article

# In[72]:


df_said = df_clean[df_clean.news.apply(lambda news_text: check_string_for('said', news_text))]


# In[73]:


df_said


# In[74]:


df_said['label'].value_counts()


# The stories containing the word said are indicative of the news story being true. With twice as many of the "true" news stories containing said vs. "fake", the true ones must seem likely to be more concerned with providing quotations, or at least quotations in this style.

# On in news article

# In[75]:


df_on = df_clean[df_clean.news.apply(lambda news_text: check_string_for(' on ', news_text))]


# In[76]:


df_on


# In[77]:


df_on.label.value_counts()


# The use of 'on' is fairly balanced although somewhat indicative of a 'true' story.

# You in news article

# In[78]:


df_you = df_clean[df_clean.news.apply(lambda news_text: check_string_for(' you ', news_text))]


# In[79]:


df_you


# In[80]:


df_you.label.value_counts()


# You is equally present in both true and fake news stories.

# Save the Cleaned Dataset

# In[81]:


df_clean.to_csv(os.path.join(dataset_dir,'train_news_cleaned.csv'),index=False)


# In[ ]:




