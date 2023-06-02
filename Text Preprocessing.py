#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys


# In[2]:


get_ipython().system('{sys.executable} -m pip install pandas')
get_ipython().system('{sys.executable} -m pip install nltk')
get_ipython().system('{sys.executable} -m pip install regex')


# In[3]:


import os
import pandas as pd
import re
import string
import nltk
from nltk.tokenize import RegexpTokenizer, word_tokenize, TweetTokenizer
from nltk import FreqDist
from nltk.corpus import stopwords


# Importing the dataset

# In[4]:


cwd = os.getcwd()
dataset_dir = os.path.join(cwd,'Dataset')
df = pd.read_csv(os.path.join(dataset_dir,'train_news_cleaned.csv'))

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
# In[5]:


print("Dataset shape:", df.shape)


# In[6]:


df.head()


# In[7]:


df.info()


# Clean Headline and News data.

# Helper Functions

# In[8]:


def concat_lists_of_strings(df, column):
    """Concatenate a series of lists of strings from a column in a dataframe"""
    return [x for list_ in df[column].values for x in list_]


# In[9]:


def find_strings(string_, regex):
    """Find and Return a list of URLs in the input string"""
    list_ = re.findall(regex, string_)
    return [s[0] for s in list_]


# In[10]:


def freq_dist_of_col(df, col):
    """Return a Frequency Distribution of a column"""
    corpus_tokens = concat_lists_of_strings(df, col)
    corpus_freq_dist = FreqDist(corpus_tokens)
    print(f'The number of unique tokens in the corpus is {len(corpus_freq_dist)}')
    return corpus_freq_dist


# In[11]:


def review_freq_dis(df, col, n):
    """
    Create a Frequency Distribution of a column of a dataframe and display
    the n most common tokens.
    """
    corpus_freq_dist = freq_dist_of_col(df, col)
    display(corpus_freq_dist.most_common(n))


# In[12]:


def remove_punctuation(word_list, punctuation_list):
    """Remove punctuation tokens from a list of tokens"""
    return [w for w in word_list if w not in punctuation_list]


# In[13]:


def remove_single_characters(word_list, exception_list):
    """Remove all the single characters, except those on the exception list"""
    return [w for w in word_list if (len(w) > 1 or w in exception_list)]


# In[14]:


def remove_words(word_list, words_to_remove):
    """Remove all the words in the words_to_remove list from the words_list"""
    return [w for w in word_list if w not in words_to_remove]


# Token Frequency Distribution

# In[15]:


tknzr = RegexpTokenizer(r'\w+|\$[\d\.]+|\([@\w\d]+\)')
df['news_tokens'] = df['news'].apply(tknzr.tokenize)


# In[16]:


corpus_freq_dist = freq_dist_of_col(df, 'news_tokens')


# In[17]:


corpus_freq_dist.most_common(150)


# Tokens used only once

# In[18]:


len([w for w in corpus_freq_dist.most_common() if w[1] == 1])


# Tokens used less than 5 times

# In[19]:


len([w for w in corpus_freq_dist.most_common() if w[1] <= 5])


# At the top of the frequency distribution, the usual stop words are present, along with with words associated with politics or the names of political figures, institutions or countries.

# The amount of words that are used only once or 5 or less times is relatively small given the size of the corpus.

# Investigate if URLs are present in the news article text

# In[20]:


URL_REGEX = r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))"


# In[21]:


df['news_urls'] = df['news'].apply(lambda x: find_strings(x, URL_REGEX))


# In[22]:


urls_in_news = concat_lists_of_strings(df, 'news_urls')


# In[23]:


urls_in_news


# In[24]:


len(urls_in_news)


# In[25]:


url_freq_dist = FreqDist(urls_in_news)


# In[26]:


url_freq_dist.most_common(150)


# The first two links are mentioned multiple times but when I tried to check them out they were sealed. As more time goes on more of the links will stop working. So it would be better to replace them with a placeholder {link}.

# In[27]:


df['clean_news'] = df['news'].apply(lambda x: re.sub(URL_REGEX, '{link}', x))


# Investigate if URLs are present in the headline text

# In[28]:


df['headline_urls'] = df['headline'].apply(lambda x: find_strings(x, URL_REGEX))


# In[29]:


urls_in_headline = concat_lists_of_strings(df, 'headline_urls')


# In[30]:


urls_in_headline


# Replacing the links in the headlines with placeholder aswell.

# In[31]:


df['clean_headline'] = df['headline'].apply(lambda x: re.sub(URL_REGEX, '{link}', x))


# Investigate Twitter handles in news articles.

# In[32]:


TWITTER_HANDLE_REGEX = r'(?<=^|(?<=[^\w]))(@\w{1,15})\b'


# In[33]:


df['twitter_handles'] = df['clean_news'].apply(lambda x: re.findall(TWITTER_HANDLE_REGEX, x))


# In[34]:


twitter_handles = concat_lists_of_strings(df, 'twitter_handles')


# In[35]:


twitter_freq_dist = FreqDist(twitter_handles)


# In[36]:


twitter_freq_dist.most_common(50)


# In[37]:


len(twitter_handles)


# In[38]:


len(twitter_freq_dist)


# In[39]:


df['clean_news'] = df['clean_news'].apply(lambda x: re.sub(TWITTER_HANDLE_REGEX, '@twitter-handle', x))


# Capitalization

# Because words with all caps are an import way that emphasis is made online, we will keep words that are in all caps while making all the letters in other words lower case. Words of length of one will be made lower case though since they are likely A or I which can be made lowercase without losing much emphasis.

# In[40]:


def lower_unless_all_caps(string_):
    """
    Make all words in the input string lowercase unless that 
    word is in all caps
    """
    words = string_.split()
    processed_words = [w.lower() if not (w.isupper() and len(w) > 1) else w for w in words]
    return ' '.join(processed_words)


# In[41]:


df['clean_news'] = df['clean_news'].apply(lower_unless_all_caps)


# In[42]:


df['clean_headline'] = df['clean_headline'].apply(lower_unless_all_caps)


# In[43]:


df.head()


# Number in data

# I will replace the numbers with a space because some of the sentences run together and end with a number. Replacing the number with a space will split the sentences.

# In[44]:


df['clean_news'] = df['clean_news'].apply(lambda x: re.sub(r'9\/11', 'nine-eleven', x))


# In[45]:


df['clean_news'] = df['clean_news'].apply(lambda x: re.sub(r'\d+', ' ', x))


# In[46]:


df['clean_headline'] = df['clean_headline'].apply(lambda x: re.sub(r'9\/11', 'nine-eleven', x))


# In[47]:


df['clean_headline'] = df['clean_headline'].apply(lambda x: re.sub(r'\d+', ' ', x))


# In[48]:


nltk.download('punkt')


# Tokens in the current clean news articles

# In[49]:


df['clean_news_tokens'] = df['clean_news'].apply(word_tokenize)


# In[50]:


review_freq_dis(df, 'clean_news_tokens', 150)


# Removing all of the Punctuation tokens except for the exclamation point, because it seems like it may be an indicator of Fake news. Also removing all the single characters except for i.

# In[51]:


df['clean_news_tokens'] = df['clean_news_tokens'].apply(lambda x: remove_single_characters(x, ['i', '!']))


# In[52]:


review_freq_dis(df, 'clean_news_tokens', 150)


# Tokens in the current clean headline

# In[53]:


df['clean_headline_tokens'] = df['clean_headline'].apply(word_tokenize)


# In[54]:


review_freq_dis(df, 'clean_headline_tokens', 150)


# Remove Punctuation and Single Letter Tokens from Clean Headline

# In[55]:


df['clean_headline_tokens'] = df['clean_headline_tokens'].apply(lambda x: remove_single_characters(x, ['i', '!']))


# In[56]:


review_freq_dis(df, 'clean_headline_tokens', 150)


# Removing "'s"

# While the fake news frequently or always didn't removed the apostrophe from 's, it doesn't look like that was done to the true news. 's will need to be removed so that it doesn't become a false indicator of true news.

# In[57]:


df['clean_headline_tokens'] = df['clean_headline_tokens'].apply(lambda x: remove_words(x, ["'s"]))
df['clean_news_tokens'] = df['clean_news_tokens'].apply(lambda x: remove_words(x, ["'s"]))


# Remove Date Words

# To better generalize the models removing all the date words.

# In[58]:


date_words = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 
              'saturday', 'sunday', 'january', 'february', 'march', 'april',
             'may', 'june', 'july', 'august', 'september', 'october',
             'november', 'december']


# In[59]:


df['clean_headline_tokens'] = df['clean_headline_tokens'].apply(lambda x: remove_words(x, date_words))
df['clean_news_tokens'] = df['clean_news_tokens'].apply(lambda x: remove_words(x, date_words))


# In[61]:


nltk.download('stopwords')


# Remove Stop Words

# In[62]:


stop_words = stopwords.words('english')


# In[63]:


display(stop_words)


# In[64]:


most_freq_clean_news = [x[0] for x in list(freq_dist_of_col(df, 'clean_news_tokens').most_common(150))]


# In[65]:


most_freq_clean_news


# In[66]:


def intersection(lst1, lst2):
    """Return the intersection of two lists"""

    temp = set(lst2) 
    lst3 = [value for value in lst1 if value in temp] 
    return lst3


# In[67]:


common_words = intersection(stop_words, most_freq_clean_news)


# In[68]:


common_words


# In[69]:


len(common_words)


# In[70]:


len(stop_words)


# In[71]:


def difference(lst1, lst2):
    """Return the difference of two lists"""

    temp = set(lst2) 
    lst3 = [value for value in lst1 if value not in temp] 
    return lst3


# In[72]:


words_in_nltk_not_news = difference(stop_words, most_freq_clean_news)


# In[73]:


words_in_nltk_not_news


# In[74]:


words_in_news_not_nltk = difference(most_freq_clean_news, stop_words)


# In[75]:


words_in_news_not_nltk


# Looking at the remaining frequent words from the news text, that are all very concentrated on political news.

# Saving data

# In[76]:


df.to_csv(os.path.join(dataset_dir,'train_news_preprocessed.csv'),index=False)


# In[ ]:




