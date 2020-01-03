#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import pandas as pd
import numpy as np
import matplotlib.ticker as ticker
from sklearn import preprocessing
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


get_ipython().system('wget -O loan_train.csv https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/loan_train.csv')


# In[ ]:


dataset = pd.read_csv('loan_train.csv')


# In[ ]:


dataset.head()


# In[ ]:


dataset.shape


# In[ ]:


dataset.describe()


# In[ ]:


drop_columns = ["Unnamed: 0", "Unnamed: 0.1"]
dataset.drop(labels= drop_columns, axis=1, inplace=True)


# In[ ]:


dataset.head()


# In[ ]:


dataset.shape


# In[ ]:


dataset.dtypes


# In[249]:


dataset['loan_status'].value_counts()


# In[250]:


dataset['education'].value_counts()


# In[251]:


dataset['Gender'].value_counts()


# In[252]:


# z = dataset.iloc[:, 0:1].values


# In[253]:


# Any text data needs to be converted into numbers that our model can use,
# We'll also fill any empty cells with 0:

# dataset = pd.get_dummies(dataset, columns=["education"])
# dataset.fillna(value=0.0, inplace=True)


# In[254]:


# Any text data needs to be converted into numbers that our model can use,
# We'll also fill any empty cells with 0:

# dataset = pd.get_dummies(dataset, columns=["Gender"])
# dataset.fillna(value=0.0, inplace=True)


# In[255]:


# Any text data needs to be converted into numbers that our model can use,
# We'll also fill any empty cells with 0:

# dataset = pd.get_dummies(dataset, columns=["loan_status"])
# dataset.fillna(value=0.0, inplace=True)


# In[256]:


dataset['due_date'] = pd.to_datetime(dataset['due_date'])
dataset['effective_date'] = pd.to_datetime(dataset['effective_date'])


# In[257]:


dataset.head()


# In[258]:


dataset.shape


# In[259]:


dataset.dtypes


# In[260]:


# notice: installing seaborn might takes a few minutes
get_ipython().system('conda install -c anaconda seaborn -y')


# In[263]:


import seaborn as sns

bins = np.linspace(dataset.Principal.min(), dataset.Principal.max(), 10)
g = sns.FacetGrid(dataset, col="Gender", hue="loan_status", palette="Set1", col_wrap=2)
g.map(plt.hist, 'Principal', bins=bins, ec="k")

g.axes[-1].legend()
plt.show()


# In[265]:


bins = np.linspace(dataset.age.min(), dataset.age.max(), 10)
g = sns.FacetGrid(dataset, col="Gender", hue="loan_status", palette="Set1", col_wrap=2)
g.map(plt.hist, 'age', bins=bins, ec="k")

g.axes[-1].legend()
plt.show()


# In[267]:


dataset['dayofweek'] = dataset['effective_date'].dt.dayofweek
bins = np.linspace(dataset.dayofweek.min(), dataset.dayofweek.max(), 10)
g = sns.FacetGrid(dataset, col="Gender", hue="loan_status", palette="Set1", col_wrap=2)
g.map(plt.hist, 'dayofweek', bins=bins, ec="k")
g.axes[-1].legend()
plt.show()


# We see that people who get the loan at the end of the week dont pay it off, so lets use Feature binarization to set a threshold values less then day 4

# In[271]:


dataset['weekend'] = dataset['dayofweek'].apply(lambda x: 1 if (x>3)  else 0)
dataset.head()


# In[272]:


dataset.shape


# # Convert Categorical features to numerical values

# In[274]:


# Lets look at gender:
dataset.groupby(['Gender'])['loan_status'].value_counts(normalize=True)


# In[276]:


# Lets convert male to 0 and female to 1:
dataset['Gender'].replace(to_replace=['male','female'], value=[0,1],inplace=True)
dataset.head()


# In[277]:


dataset['Gender'] = dataset['Gender'].astype('float')
dataset.head()


# In[278]:


dataset['Gender'] = dataset['Gender'].astype('int')
dataset.head()


# # One Hot Encoding

# How about education?

# In[280]:


dataset.groupby(['education'])['loan_status'].value_counts(normalize=True)

Feature before One Hot Encoding
# In[283]:


dataset[['Principal','terms','age','Gender','education']].head()


# Use one hot encoding technique to conver categorical varables to binary variables and append them to the feature Data Frame

# In[285]:


Feature = dataset[['Principal','terms','age','Gender','weekend']]
Feature = pd.concat([Feature,pd.get_dummies(dataset['education'])], axis=1)
Feature.drop(['Master or Above'], axis = 1,inplace=True)
Feature.head()


# # Feature selection

# Lets defind feature sets, X:

# In[286]:


X = Feature
X[0:5]


# In[287]:


y = dataset['loan_status'].values
y[0:5]


# # Normalize Data

# Data Standardization give data zero mean and unit variance (technically should be done after train test split )

# In[289]:


X= preprocessing.StandardScaler().fit(X).transform(X)
X[0:5]


# # K Nearest Neighbor(KNN)

# In[329]:


# Train Test Split

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20,random_state=4)


# In[330]:


# Preprocessing

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


# In[331]:


from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5)
classifier.fit(X_train, y_train)


# In[334]:


y_pred = classifier.predict(X_test)


# In[335]:


from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(accuracy_score(y_test, y_pred))


# In[336]:


error = []

# Calculating error for K values between 1 and 40
for i in range(1, 40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error.append(np.mean(pred_i != y_test))


# In[337]:


plt.figure(figsize=(12, 6))
plt.plot(range(1, 40), error, color='red', linestyle='dashed', marker='o',
         markerfacecolor='blue', markersize=10)
plt.title('Error Rate K Value')
plt.xlabel('K Value')
plt.ylabel('Mean Error')


# In[338]:


from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 7)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)


# In[339]:


from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(accuracy_score(y_test, y_pred))


# # Decision Tree

# The X and y are the arrays required before the split, the test_size represents the ratio of the testing dataset, and the random_state ensures that we obtain the same splits.

# In[397]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10)


# Inside of the classifier, specify criterion="entropy" so we can see the information gain of each node.

# In[398]:


from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion="entropy", max_depth = 2)
classifier # it shows the default parameters


# Next, we will fit the data with the training feature matrix X_trainset and training response vector y_trainset

# In[399]:


classifier.fit(X_train,y_train)


# # Prediction

# Let's make some predictions on the testing dataset and store it into a variable called y_pred

# In[400]:


y_pred = classifier.predict(X_test)


# # Evaluation

# Next, let's import metrics from sklearn and check the accuracy of our model.

# In[401]:


from sklearn import metrics
import matplotlib.pyplot as plt
print("DecisionTrees's Accuracy: ", metrics.accuracy_score(y_test, y_pred))


# In[ ]:





# # Support Vector Machine

# In[ ]:


#  the model_selection library of the Scikit-Learn library contains the train_test_split
# method that allows us to seamlessly divide data into training and test sets.

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20,random_state = 10)


# In[ ]:


# Training the Algorithm

#from sklearn.svm import SVC   # Support Vector Classifier
#svclassifier = SVC(kernel='linear')#This class takes one parameter,which is the kernel type
#svclassifier.fit(X_train, y_train)


# In[ ]:


#y_pred = svclassifier.predict(X_test)


# In[ ]:


#from sklearn.metrics import classification_report, confusion_matrix,accuracy_score
#print(confusion_matrix(y_test,y_pred))
#print(classification_report(y_test,y_pred))
#print(accuracy_score(y_test, y_pred))


# In[ ]:





# # Kernel SVM

#  Polynomial Kernel

# In[ ]:


# Polynomial Kernel
from sklearn.svm import SVC   # Support Vector Classifier
classifier = SVC(kernel ='poly', degree= 8)
classifier.fit(X_train, y_train)


# In[ ]:


y_pred = svclassifier.predict(X_test)


# In[ ]:


# Evaluating the Algorithm
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(accuracy_score(y_test, y_pred))


# In[ ]:





# In[ ]:





# In[ ]:




