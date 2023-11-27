#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report
from sklearn.metrics import r2_score
from sklearn.metrics import confusion_matrix


# In[2]:

def allmodel(x_train, x_test, y_train, y_test):
    models = {"Logistic Regression" : LogisticRegression(),
              "Decision Tree" : DecisionTreeClassifier(),
              "Random Forest" : RandomForestClassifier(),
              "Naive bayes" : GaussianNB()}
    for i in range(len(list(models))):
        print("For",list(models.keys())[i])
        model = list(models.values())[i]
        model.fit(x_train,y_train)
    
        y_train_pred = model.predict(x_train)
        y_test_pred = model.predict(x_test)
        
        return y_train_pred, y_test_pred

def Best_Model(x_train, x_test, y_train, y_test):
        y_train_pred,y_test_pred = allmodel(x_train, x_test, y_train, y_test)
    
        ## train set accurecy
        print("train set classification_report\n")
        print(classification_report(y_train, y_train_pred))
        print("test set classification_report\n")
        print(classification_report(y_test, y_test_pred))
        print("==============================================================================")


# In[3]:


def Accuracy(x_train, x_test, y_train, y_test):
    y_train_pred,y_test_pred = allmodel(x_train, x_test, y_train, y_test)
    
        ## train set accurecy
        print("train set accurecy",r2_score(y_train, y_train_pred)*100)
        ## test set accurecy
        print("test set accurecy",r2_score(y_test, y_test_pred)*100)
        print("==============================================================================")


# In[4]:


def Confusion_matrix(x_train, x_test, y_train, y_test):
    y_train_pred,y_test_pred = allmodel(x_train, x_test, y_train, y_test)
    
        ## train set accurecy
        print("train set confusion_matrix\n")
        print(confusion_matrix(y_train, y_train_pred)*100)
        ## test set accurecy
        print("test set confusion_matrix")
        print(confusion_matrix(y_test, y_test_pred)*100)
        print("==============================================================================")


# In[ ]:




