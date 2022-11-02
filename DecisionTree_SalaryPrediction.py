#!/usr/bin/env python
# coding: utf-8

# # Decision Tree: Income Prediction
# In this lab, we will build a decision tree to predict the income of a given population, which is labelled as <= 50k 𝑎𝑛𝑑 > 50k. The attributes (predictors) are age, working class type, marital status, gender, race etc.
# 
# In the following sections, we'll:
# 
# Clean and prepare the data,
# - build a decision tree with default hyperparameters,
# - understand all the hyperparameters that we can tune, and finally
# - choose the optimal hyperparameters using grid search cross-validation.

# In[1]:


# Importing the required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# Reading the csv file and putting it into 'df' object.
df = pd.read_csv('adult_dataset.csv')


# In[3]:


# Let's understand the data, how it look like.
df.head()


# # NOTE
# This dataset contains missing rows with a value='?'. Remove the missing values by dropping those rows.

# In[4]:


# select all categorical variables
df_categorical = df.select_dtypes(include=['object'])


# In[5]:


# checking whether any other columns contain a "?"
df_categorical.apply(lambda x: x=="?", axis=0).sum()


# In[6]:


# dropping the "?"s
df = df[df['workclass'] != '?']
df = df[df['occupation'] != '?']
df = df[df['native.country'] != '?']


# In[7]:


# clean dataframe
df.info()


# # Data Preparation
# There are a number of preprocessing steps we need to do before building the model.
# 
# Firstly, note that we have both categorical and numeric features as predictors. In previous models such as linear and logistic regression, we had created dummy variables for categorical variables, since those models (being mathematical equations) can process only numeric variables.
# 
# All that is not required in decision trees, since they can process categorical variables easily. However, we still need to encode the categorical variables into a standard format so that sklearn can understand them and build the tree. We'll do that using the LabelEncoder() class, which comes with sklearn.preprocessing.

# In[8]:


#import required libraries
from sklearn import preprocessing


# In[9]:


# select all categorical variables from the clean dataframe
df_categorical = df.select_dtypes(include=['object'])
df_categorical.head()


# In[10]:


# apply Label encoder to df_categorical
le = preprocessing.LabelEncoder()
df_categorical = df_categorical.apply(le.fit_transform)
df_categorical.head()


# In[11]:


# concat df_categorical with original df
df = df.drop(df_categorical.columns, axis=1)
df = pd.concat([df, df_categorical], axis=1)
df.head()


# In[12]:


# convert target variable income to categorical
df['income'] = df['income'].astype('category')


# Now all the categorical variables are suitably encoded. Let's build the model.

# # Model Building and Evaluation
# Let's first build a decision tree with default hyperparameters. Then we'll use cross-validation to tune them.

# In[13]:


# Importing train-test-split 
from sklearn.model_selection import train_test_split


# In[14]:


# Putting feature variable to X
X = df.drop('income',axis=1)

# Putting response variable to y
y = df['income']


# In[15]:


# Splitting the data into train and test (70/30 ratio)
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.30, 
                                                    random_state = 99)
X_train.head()


# In[16]:


# Importing decision tree classifier from sklearn library
from sklearn.tree import DecisionTreeClassifier


# In[17]:


# Build a Decision Tree
dt_default = DecisionTreeClassifier()
dt_default.fit(X_train, y_train)


# In[18]:


# Let's check the evaluation metrics of our default model

# Importing classification report and confusion matrix from sklearn metrics
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Making predictions
y_pred_default = dt_default.predict(X_test)

# Printing classification report
print(classification_report(y_test, y_pred_default))


# In[19]:


# Printing confusion matrix and accuracy
print(confusion_matrix(y_test,y_pred_default))
print(accuracy_score(y_test,y_pred_default))


# # Plotting the Decision Tree
# 
# To visualise decision trees in python, you need to install certain external libraries. You can read about the process in detail here: http://scikit-learn.org/stable/modules/tree.html
# 
# We need the ```graphviz``` library to plot a tree.

# In[20]:


# Importing required packages for visualization
from IPython.display import Image  
from sklearn.externals.six import StringIO  
from sklearn.tree import export_graphviz
import pydotplus, graphviz


# In[21]:


# Putting features
features = list(df.columns[1:])
features


# In[22]:


import os
os.environ["PATH"] += os.pathsep + 'C:\Program Files (x86)\Graphviz2.38\bin'


# In[23]:


dot_data = StringIO()  
export_graphviz(dt_default, out_file=dot_data,
                feature_names=features, filled=True,rounded=True)

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())


# # OPTIMAL HYPERPARAMETERS

# In[24]:


#import libraries required
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV


# In[25]:


# Create the parameter grid 
param_grid = {
    'max_depth': range(5, 15, 5),
    'min_samples_leaf': range(50, 150, 50),
    'min_samples_split': range(50, 150, 50),
    'criterion': ["entropy", "gini"]
}


# In[26]:


n_folds = 5


# In[27]:


# Instantiate the grid search model
dtree = DecisionTreeClassifier()
grid_search = GridSearchCV(estimator = dtree, param_grid = param_grid, 
                          cv = n_folds, verbose = 1)

# Fit the grid search to the data
grid_search.fit(X_train,y_train)


# In[28]:


# printing the optimal accuracy score and hyperparameters
print("best accuracy", grid_search.best_score_)
print(grid_search.best_estimator_)


# # Running the model with best parameters obtained from grid search.

# In[29]:


# model with optimal hyperparameters
clf_gini = DecisionTreeClassifier(criterion = "gini", 
                                  random_state = 100,
                                  max_depth=10, 
                                  min_samples_leaf=50,
                                  min_samples_split=50)
clf_gini.fit(X_train, y_train)


# In[30]:


# accuracy score
accuracy=clf_gini.score(X_test,y_test)
accuracy


# In[31]:


# plotting the tree
dot_data = StringIO()  
export_graphviz(clf_gini, out_file=dot_data,feature_names=features,filled=True,rounded=True)

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())

