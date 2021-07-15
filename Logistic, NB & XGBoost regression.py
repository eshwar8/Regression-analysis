#!/usr/bin/env python
# coding: utf-8

# In[1]:


import xgboost as xg
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
from sklearn import datasets
import statsmodels.api as sm
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import cross_val_score 
import statsmodels.formula.api as smf
import seaborn as sns
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB


# In[2]:


Exam_2=pd.read_csv("Education regression dataset.csv")


# In[3]:


print(Exam_2.head())
Exam_2.dtypes


# In[4]:


sns.countplot(x='Good or Bad',data=Exam_2,palette='hls')


# # Deciding the variables

# In[5]:


Feature_cols = ['Gender', 'Under Graduation', '12th in percent','10th in percent','Work Experience in months']
X = Exam_2[Feature_cols]
Y = Exam_2['Good or Bad']


# # Train Test Split

# In[6]:


seed = 7
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.25,random_state=0)


# # Logistic Regression model

# In[7]:


model3 = sm.Logit(Y_train, X_train).fit()


# In[8]:


model3.summary()


# In[9]:


y_pred3 = model3.predict(X_test)
y_pred3


# In[10]:


accuracy3 = accuracy_score(Y_test,round(y_pred3,0))
accuracy3


# In[11]:


print(classification_report(Y_test,round(y_pred3,0)))


# # XGboost Model

# In[12]:


model4 = XGBClassifier()
model4.fit(X_train,Y_train)


# In[13]:


y_pred4 = model4.predict(X_test)
y_pred4


# In[14]:


accuracy4 = accuracy_score(Y_test,y_pred4)
accuracy4


# In[15]:


print(classification_report(Y_test,y_pred4))


# # Multinomial Naive Bayes model

# In[16]:


clf = MultinomialNB().fit(X_train, Y_train)
predicted= clf.predict(X_test)
print(classification_report(Y_test, predicted))


# # Result: Referring to the accuracy, Logistic model is the suitable model for this data followed by XGboost. 

# In[ ]:





# In[ ]:




