#!/usr/bin/env python
# coding: utf-8

# # FAST AND FURIOUS First TRAIN DATA

# In[2]:


# Import pandas
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


# In[3]:


from sklearn.metrics import classification_report


# In[4]:


data=pd.read_csv('THE-FAST-AND-THE-FURIOUS-2.csv', sep='\t')


# # Remove Punctuations

# In[5]:


data.Comments = data.Comments.str.replace('[^\w\s]', '')


# In[6]:


data.head()


# In[7]:


data["Comments"]


# # Polarity is a float that lies between [-1,1], -1 indicates negative sentiment and +1 indicates positive sentiments &                                                                           Subjectivity is also a float that lies in the range of [0,1]. Subjective sentences generally refer to opinion, emotion, or judgment. So Subjectivity will be used as a sentiment review for training purpose.

# In[8]:


pol = lambda x: TextBlob(x).sentiment.polarity
sub = lambda x: TextBlob(x).sentiment.subjectivity

data['polarity'] = data['Comments'].apply(pol)
data['subjectivity']=data['Comments'].apply(sub)
data


# In[9]:


train_data= data


# In[10]:


train_data.polarity=round(train_data.polarity,)
train_data.subjectivity=round(train_data.subjectivity,)


# In[11]:


train_data.head()


# In[12]:


train_data=train_data.rename(columns=({'polarity':'Sentiment_1'}))
train_data=train_data.rename(columns=({'subjectivity':'Sentiment_2'}))


# In[13]:


train_data.head()


# In[14]:


print(train_data.Sentiment_1.value_counts())
print(train_data.Sentiment_2.value_counts())


# In[15]:


sns.distplot(train_data['Sentiment_1'])


# In[16]:


sns.distplot(train_data['Sentiment_2'])


# # Tracing the Most negative sentences using polarity & subjectivity

# In[17]:


most_negative_polarity_1 = train_data[train_data.Sentiment_1==-1].Comments
print(most_negative_polarity_1)
print(len(most_negative_polarity_1))


# In[18]:


most_negative_subjectivity_1=train_data[train_data.Sentiment_2<1].Comments
print(most_negative_subjectivity_1)
print(len(most_negative_subjectivity_1))


# # Training the model with Naive bayes

# In[19]:


#tokenizer to remove unwanted elements from out data like symbols and numbers
token = RegexpTokenizer(r'[a-zA-Z0-9]+')
cv = CountVectorizer(lowercase=True,stop_words='english',ngram_range = (1,1),tokenizer = token.tokenize)
text_counts= cv.fit_transform(train_data['Comments'])


# In[20]:


X_train, X_test, y_train, y_test = train_test_split(
    text_counts, train_data['Sentiment_2'], test_size=0.3, random_state=1)


# In[21]:


clf = MultinomialNB().fit(X_train, y_train)
predicted= clf.predict(X_test)
print("MultinomialNB Accuracy:",metrics.accuracy_score(y_test, predicted))


# # Training the model with Logisitic Regression

# In[22]:


classifier = LogisticRegression()
classifier.fit(X_train, y_train)


# In[23]:


accuracy = classifier.score(X_test, y_test)
print('Accuracy:', accuracy)


# In[ ]:





# In[ ]:





# In[ ]:





# # MISSION IMPOSSIBLE 1996 Second TRAIN DATA 

# In[24]:


train_data_2=pd.read_csv('mission-impossible-1996.csv', sep='\t')


# In[25]:


train_data_2.head()


# # Remove Punctuations

# In[26]:


train_data_2.Comments = train_data_2.Comments.str.replace('[^\w\s]', '')


# In[27]:


pol_2 = lambda x: TextBlob(x).sentiment.polarity
sub_2 = lambda x: TextBlob(x).sentiment.subjectivity

train_data_2['polarity'] = train_data_2['Comments'].apply(pol_2)
train_data_2['subjectivity']=train_data_2['Comments'].apply(sub_2)
train_data_2


# In[28]:


train_data_2.polarity=round(train_data_2.polarity,)
train_data_2.subjectivity=round(train_data_2.subjectivity,)


# In[29]:


train_data_2=train_data_2.rename(columns=({'polarity':'Sentiment_1'}))
train_data_2=train_data_2.rename(columns=({'subjectivity':'Sentiment_2'}))


# In[30]:


train_data_2.head()


# In[31]:


print(train_data_2.Sentiment_1.value_counts())
print(train_data_2.Sentiment_2.value_counts())


# In[32]:


sns.distplot(train_data_2['Sentiment_1'])


# In[33]:


sns.distplot(train_data_2['Sentiment_2'])


# # Tracing the Most negative sentences using polarity and subjectivity

# In[34]:


most_negative_2_polarity = train_data_2[train_data_2.Sentiment_1==-1].Comments
print(most_negative_2_polarity)
print(len(most_negative_2_polarity))


# In[35]:


most_negative_2_subjectivity=train_data_2[train_data_2.Sentiment_2<1].Comments
print(most_negative_2_subjectivity)
print(len(most_negative_2_subjectivity))


# # Training the model with Naive bayes

# In[36]:


token = RegexpTokenizer(r'[a-zA-Z0-9]+')
cv = CountVectorizer(lowercase=True,stop_words='english',ngram_range = (1,1),tokenizer = token.tokenize)
text_counts= cv.fit_transform(train_data_2['Comments'])


# In[37]:


X_train2, X_test2, y_train2, y_test2 = train_test_split(
    text_counts, train_data_2['Sentiment_2'], test_size=0.3, random_state=1)


# In[38]:


clf = MultinomialNB().fit(X_train2, y_train2)
predicted2= clf.predict(X_test2)
print("MultinomialNB Accuracy II:",metrics.accuracy_score(y_test2, predicted2))


# # Training the model with Logisitic Regression

# In[39]:


classifier2 = LogisticRegression()
classifier2.fit(X_train2, y_train2)


# In[40]:


accuracy2 = classifier2.score(X_test2, y_test2)
print('Accuracy2:', accuracy2)


# In[ ]:





# In[ ]:





# In[ ]:





# # Rambo 3 Third TRAIN DATA

# In[41]:


train_data_3=pd.read_csv('Rambo_III.csv', sep='\t')


# # Remove Punctuations

# In[42]:


train_data_3.Comments = train_data_3.Comments.str.replace('[^\w\s]', '')


# In[43]:


train_data_3.head()


# In[44]:


pol_3 = lambda x: TextBlob(x).sentiment.polarity
sub_3 = lambda x: TextBlob(x).sentiment.subjectivity

train_data_3['polarity'] = train_data_3['Comments'].apply(pol_3)
train_data_3['subjectivity']=train_data_3['Comments'].apply(sub_3)
train_data_3


# In[45]:


train_data_3.polarity=round(train_data_3.polarity,)
train_data_3.subjectivity=round(train_data_3.subjectivity,)


# In[46]:


train_data_3=train_data_3.rename(columns=({'polarity':'Sentiment_1'}))
train_data_3=train_data_3.rename(columns=({'subjectivity':'Sentiment_2'}))


# In[47]:


train_data_3.head()


# In[48]:


print(train_data_3.Sentiment_1.value_counts())
print(train_data_3.Sentiment_2.value_counts())


# In[49]:


sns.distplot(train_data_3['Sentiment_1'])


# In[50]:


sns.distplot(train_data_3['Sentiment_2'])


# # Tracing the Most negative sentences using polarity & Subjectivity

# In[51]:


most_negative_3_polarity = train_data_3[train_data_3.Sentiment_1==-1].Comments
print(most_negative_3_polarity)
print(len(most_negative_3_polarity))


# In[52]:


most_negative_3_subjectivity = train_data_3[train_data_3.Sentiment_1==-1].Comments
print(most_negative_3_subjectivity)
print(len(most_negative_3_subjectivity))


# # Training the model with Naive bayes

# In[53]:


token = RegexpTokenizer(r'[a-zA-Z0-9]+')
cv = CountVectorizer(lowercase=True,stop_words='english',ngram_range = (1,1),tokenizer = token.tokenize)
text_counts= cv.fit_transform(train_data_3['Comments'])


# In[54]:


X_train3, X_test3, y_train3, y_test3 = train_test_split(
    text_counts, train_data_3['Sentiment_2'], test_size=0.3, random_state=1)


# In[55]:


clf = MultinomialNB().fit(X_train3, y_train3)
predicted3= clf.predict(X_test3)
print("MultinomialNB Accuracy III:",metrics.accuracy_score(y_test3, predicted3))


# # Training the model with Logisitic Regression

# In[56]:


classifier3 = LogisticRegression()
classifier3.fit(X_train3, y_train3)


# In[57]:


accuracy3 = classifier3.score(X_test3, y_test3)
print('Accuracy3:', accuracy3)


# In[ ]:





# In[ ]:





# In[ ]:





# # The Dark night Arises Fourth TRAIN DATA

# In[58]:


train_data_4=pd.read_csv('The-dark-knight-rises.csv', sep='\t')


# # Remove Punctuations

# In[59]:


train_data_4.Comments = train_data_4.Comments.str.replace('[^\w\s]', '')


# In[60]:


pol_4 = lambda x: TextBlob(x).sentiment.polarity
sub_4 = lambda x: TextBlob(x).sentiment.subjectivity

train_data_4['polarity'] = train_data_4['Comments'].apply(pol_4)
train_data_4['subjectivity']=train_data_4['Comments'].apply(sub_4)
train_data_4


# In[61]:


train_data_4.head()


# In[62]:


train_data_4.polarity=round(train_data_4.polarity,)
train_data_4.subjectivity=round(train_data_4.subjectivity,)


# In[63]:


train_data_4=train_data_4.rename(columns=({'polarity':'Sentiment_1'}))
train_data_4=train_data_4.rename(columns=({'subjectivity':'Sentiment_2'}))


# In[64]:


train_data_4.head()


# In[65]:


print(train_data_4.Sentiment_1.value_counts())
print(train_data_4.Sentiment_2.value_counts())


# # Tracing the Most negative sentences using polarity & Subjectivity

# In[66]:


most_negative_4_polarity = train_data_4[train_data_4.Sentiment_1==-1].Comments
print(most_negative_4_polarity)
print(len(most_negative_4_polarity))


# In[67]:


most_negative_4_subjectivity = train_data_4[train_data_4.Sentiment_1<1].Comments
print(most_negative_4_subjectivity)
print(len(most_negative_4_subjectivity))


# # Training the model with Naive bayes

# In[68]:


token = RegexpTokenizer(r'[a-zA-Z0-9]+')
cv = CountVectorizer(lowercase=True,stop_words='english',ngram_range = (1,1),tokenizer = token.tokenize)
text_counts= cv.fit_transform(train_data_4['Comments'])


# In[69]:


X_train4, X_test4, y_train4, y_test4 = train_test_split(
    text_counts, train_data_4['Sentiment_2'], test_size=0.3, random_state=1)


# In[70]:


clf = MultinomialNB().fit(X_train4, y_train4)
predicted4= clf.predict(X_test4)
print("MultinomialNB Accuracy IV:",metrics.accuracy_score(y_test4, predicted4))


# # Training the model with Logisitic Regression

# In[71]:


classifier4 = LogisticRegression()
classifier4.fit(X_train4, y_train4)


# In[72]:


accuracy4 = classifier4.score(X_test4, y_test4)
print('Accuracy4:', accuracy4)


# In[ ]:





# In[ ]:





# In[ ]:





# # MR and MRS Smith Fifth TRAINING DATA

# In[73]:


train_data_5=pd.read_csv('mr-and-mrs-smith.csv', sep='\t')


# # Remove Punctuations

# In[74]:


train_data_5.Comments = train_data_5.Comments.str.replace('[^\w\s]', '')


# In[75]:


pol_5 = lambda x: TextBlob(x).sentiment.polarity
sub_5 = lambda x: TextBlob(x).sentiment.subjectivity

train_data_5['polarity'] = train_data_5['Comments'].apply(pol_5)
train_data_5['subjectivity']=train_data_5['Comments'].apply(sub_5)
train_data_5


# In[76]:


train_data_5.head()


# In[77]:


train_data_5.polarity=round(train_data_5.polarity,)
train_data_5.subjectivity=round(train_data_5.subjectivity,)


# In[78]:


train_data_5=train_data_5.rename(columns=({'polarity':'Sentiment_1'}))
train_data_5=train_data_5.rename(columns=({'subjectivity':'Sentiment_2'}))


# In[79]:


train_data_5.head()


# In[80]:


print(train_data_5.Sentiment_1.value_counts())
print(train_data_5.Sentiment_2.value_counts())


# # Tracing the Most negative sentences using polarity & subjectivity

# In[81]:


most_negative_5_polarity = train_data_5[train_data_5.Sentiment_1==-1].Comments
print(most_negative_5_polarity)
print(len(most_negative_5_polarity))


# In[82]:


most_negative_5_Subjectivity = train_data_5[train_data_5.Sentiment_2<1].Comments
print(most_negative_5_Subjectivity)
print(len(most_negative_5_Subjectivity))


# # Training the model with Naive bayes

# In[83]:


token = RegexpTokenizer(r'[a-zA-Z0-9]+')
cv = CountVectorizer(lowercase=True,stop_words='english',ngram_range = (1,1),tokenizer = token.tokenize)
text_counts= cv.fit_transform(train_data_5['Comments'])


# In[84]:


X_train5, X_test5, y_train5, y_test5 = train_test_split(
    text_counts, train_data_5['Sentiment_2'], test_size=0.3, random_state=1)


# In[85]:


clf = MultinomialNB().fit(X_train5, y_train5)
predicted5= clf.predict(X_test5)
print("MultinomialNB Accuracy V:",metrics.accuracy_score(y_test5, predicted5))


# # Training the model with Logisitic Regression

# In[86]:


classifier5 = LogisticRegression()
classifier5.fit(X_train5, y_train5)


# In[87]:


accuracy5 = classifier5.score(X_test5, y_test5)
print('Accuracy5:', accuracy5)


# In[ ]:





# In[ ]:





# In[ ]:





# # NB and LR RESULTS OF ALL TRAINING ACCURACIES

# In[88]:


print("Fast and furious MultinomialNB Accuracy:",round(metrics.accuracy_score(y_test, predicted)*100))
print('Fast and furious Accuracy:', round(accuracy*100))
print()
print("Mission Impossible MultinomialNB Accuracy:",round(metrics.accuracy_score(y_test2, predicted2)*100))
print('Mission Impossible Accuracy:', round(accuracy2*100))
print()
print("Rambo III MultinomialNB Accuracy:",round(metrics.accuracy_score(y_test3, predicted3)*100))
print('Rambo III Accuracy:', round(accuracy3*100))
print()
print("Dark knight rises MultinomialNB Accuracy:",round(metrics.accuracy_score(y_test4, predicted4)*100))
print('Dark knight rises Accuracy:',round(accuracy4*100))
print()
print("Mr and Mrs Smith MultinomialNB Accuracy:",round(metrics.accuracy_score(y_test5, predicted5)*100))
print('Mr and Mrs Smith Accuracy:', round(accuracy5*100))


# ## Assessing F1 Score, Precision, recall,support

# In[89]:


# Precision measures what proportion of predicted positive label is actually positive.
# Recall measures what proportion of actual positive label is correctly predicted as positive.
# F1-score is another one of the good performance metrics which leverages both precision and recall metrics. 
# F1-score can be obtained by simply taking ‘Harmonic Mean’ of precision and recall. 
# Unlike precision which mostly focuses on false-positive and recall which mostly focuses on false-negative, 
# F1-score focuses on both false positive and false negative.
# 0.0 indicates negative sentiments and 1.0 indicates positive sentiments

print(classification_report(y_test, predicted))
print(classification_report(y_test2, predicted2))
print(classification_report(y_test3, predicted3))
print(classification_report(y_test4, predicted4))
print(classification_report(y_test5, predicted5))


# In[90]:


A=print("Fast & Furious Polarity:",(len(most_negative_polarity_1)))
B=print("Fast & Furious Subjectivity:",len(most_negative_subjectivity_1))
print()
C=print("Mission Impossible Polarity:",len(most_negative_2_polarity))
D=print("Mission Impossible Subjectivity:",len(most_negative_2_subjectivity))
print()
E=print("Rambo III Polarity:",len(most_negative_3_polarity))
F=print("Rambo III Subjectivity:",len(most_negative_3_subjectivity))
print()
G=print("The Dark knight Arises Polarity:",len(most_negative_4_polarity))
H=print("The Dark knight Arises Subjectivity:",len(most_negative_4_subjectivity))
print()
I=print("Mr and Mrs Smith Polarity:",len(most_negative_5_polarity))
F=print("Mr and Mrs Smith Subjectivity:",len(most_negative_5_Subjectivity))


# In[91]:


d= {'polarity1':train_data['Sentiment_1'],'polarity2':train_data_2['Sentiment_1'],'polarity3':train_data_3['Sentiment_1'],'polarity4':train_data_4['Sentiment_1'],'polarity5':train_data_5['Sentiment_1']}
Polarity_Average= pd.DataFrame(data=d)


# In[92]:


Polarity_Average['Average polarity']= Polarity_Average.mean(axis=1)


# In[93]:


Polarity_Average


# In[94]:


Polarity_Average.head()


# In[95]:


f={'subjectivity1':train_data['Sentiment_2'],'subjectivity2':train_data_2['Sentiment_2'],'subjectivity3':train_data_3['Sentiment_2'],'subjectivity4':train_data_4['Sentiment_2'],'subjectivity5':train_data_5['Sentiment_2']}
Subjectivity_Average = pd.DataFrame(data=f)


# In[96]:


Subjectivity_Average['Average subjectivity']= Subjectivity_Average.mean(axis=1)


# In[97]:


Subjectivity_Average

