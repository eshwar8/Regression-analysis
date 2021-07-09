#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer #tokenizer to remove unwanted elements from out data like symbols and numbers #tokenizer to remove unwanted elements from out data like symbols and numbers
from nltk.tokenize import RegexpTokenizer #tokenizer to remove unwanted elements from out data like symbols and numbers
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB # Model Generation Using Multinomial Naive Bayes
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from sklearn.metrics import classification_report
import PyPDF2


# In[2]:


A=pd.read_csv("Fast_furious.csv",sep="\t")
A.head(100)


# # Remove Punctuations

# In[3]:


A.Comments = A.Comments.str.replace('[^\w\s]', '')


# In[4]:


train_data = A


# In[5]:


train_data.head()


# In[7]:


L = len(train_data)
train_index = int(0.60 * L)


# In[8]:


# split the data into a train and test data
train, test = train_data[:train_index], train_data[train_index: ]


# In[9]:


train


# # Polarity is a float that lies between [-1,1], -1 indicates negative sentiment and +1 indicates positive sentiments & Subjectivity is also a float that lies in the range of [0,1]. Subjective sentences generally refer to opinion, emotion, or judgment. So Polarity will be used as a sentiment review for training purpose.

# In[10]:


pol = lambda x: TextBlob(x).sentiment.polarity
sub = lambda x: TextBlob(x).sentiment.subjectivity

train['polarity'] = train['Comments'].apply(pol)
train['subjectivity']=train['Comments'].apply(sub)
train


# # Renaming the columns to Sentiment_1 and Sentiment _2

# In[11]:


train=train.rename(columns=({'polarity':'Sentiment_1'}))
train=train.rename(columns=({'subjectivity':'Sentiment_2'}))
train.head()


# # Tracing the Most negative sentences using polarity & subjectivity

# In[12]:


most_negative_polarity_1 = train[train.Sentiment_1==-1].Comments
print(most_negative_polarity_1)
print()
print(len(most_negative_polarity_1))


# In[14]:


from textblob.classifiers import NaiveBayesClassifier
from textblob.sentiments import NaiveBayesAnalyzer
from textblob import Blobber


# In[15]:


train = train[['Comments', 'Sentiment_1']]


# In[19]:


train_revised = train
test_revised = test


# In[18]:


train_revised = train_revised[['Comments', 'Sentiment_1']].values.tolist()


# In[21]:


train_revised


# In[22]:


test_revised = test_revised[['Comments']].values.tolist()


# In[23]:


test_revised


# In[24]:


cl = NaiveBayesClassifier(train_revised)
cl.accuracy(test_revised)

