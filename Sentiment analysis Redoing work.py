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


# In[22]:


A=pd.read_csv("Fast_furious.csv",sep="\t")
A.head(100)


# # Remove Punctuations

# In[24]:


A.Comments = A.Comments.str.replace('[^\w\s]', '')


# In[25]:


A.head()


# In[ ]:





# In[2]:


A= pd.read_csv("Fast_furious.csv",sep="\t")


# In[3]:


A.head(100)


# In[5]:


A.head()


# In[26]:


A["Comments"]


# # Polarity is a float that lies between [-1,1], -1 indicates negative sentiment and +1 indicates positive sentiments & Subjectivity is also a float that lies in the range of [0,1]. Subjective sentences generally refer to opinion, emotion, or judgment. So Polarity will be used as a sentiment review for training purpose.

# In[27]:


pol = lambda x: TextBlob(x).sentiment.polarity
sub = lambda x: TextBlob(x).sentiment.subjectivity

A['polarity'] = A['Comments'].apply(pol)
A['subjectivity']=A['Comments'].apply(sub)
A


# # Renaming the data as train data

# In[28]:


train_data = A


# In[9]:


#train_data.polarity=round(train_data.polarity,)
#train_data.subjectivity=round(train_data.subjectivity,)


# In[29]:


train_data.head()


# # Renaming the columns to Sentiment_1 and Sentiment _2

# In[30]:


train_data=train_data.rename(columns=({'polarity':'Sentiment_1'}))
train_data=train_data.rename(columns=({'subjectivity':'Sentiment_2'}))
train_data.head()


# In[31]:


print(train_data.Sentiment_1.value_counts())
print(train_data.Sentiment_2.value_counts())


# # Tracing the Most negative sentences using polarity & subjectivity

# In[32]:


most_negative_polarity_1 = train_data[train_data.Sentiment_1==-1].Comments
print(most_negative_polarity_1)
print()
print(len(most_negative_polarity_1))


# In[33]:


from textblob.classifiers import NaiveBayesClassifier


# In[34]:


sample_reviews = train_data[['Comments']].sample(1000)


# In[35]:


num_bins = 50
plt.figure(figsize=(10,6))
n, bins, patches = plt.hist(train_data.Sentiment_1, num_bins, facecolor='blue', alpha=0.5)
plt.xlabel('Polarity')
plt.ylabel('Count')
plt.title('Histogram of polarity')
plt.show()


# In[36]:


from textblob.classifiers import NaiveBayesClassifier
from textblob.sentiments import NaiveBayesAnalyzer
from textblob import Blobber


# In[37]:


train_data = train_data[['Comments', 'Sentiment_1']]


# In[38]:


L = len(train_data)
train_index = int(0.60 * L)


# In[39]:


# split the data into a train and test data
train, test = train_data[:train_index], train_data[train_index: ]


# In[40]:


print(train)
len(train)


# In[41]:


train['Sentiment_1']


# In[42]:


import seaborn as sns


# In[43]:


sns.distplot(train['Sentiment_1'])


# In[44]:


print(train.describe())
print(train.dtypes)
print(train.info())


# In[45]:


print(test)
len(test)


# # Convert the data into a list to get the accuracy

# In[46]:


train_data_1=train_data


# In[47]:


train_data_1 = train_data_1[['Comments', 'Sentiment_1']].values.tolist()


# In[48]:


train_1, test_1 = train_data_1[:train_index], train_data_1[train_index: ]


# In[50]:


cl = NaiveBayesClassifier(train_1)
cl.accuracy(test_1)


# In[ ]:




