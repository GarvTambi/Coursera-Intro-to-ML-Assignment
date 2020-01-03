#!/usr/bin/env python
# coding: utf-8

# In[801]:


import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import pandas as pd
import numpy as np
import matplotlib.ticker as ticker
from sklearn import preprocessing
get_ipython().run_line_magic('matplotlib', 'inline')


# In[802]:


get_ipython().system('wget -O loan_train.csv https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/loan_train.csv')


# In[803]:


dataset = pd.read_csv('loan_train.csv')


# In[804]:


dataset.head()


# In[805]:


dataset.shape


# In[806]:


dataset.describe()


# In[807]:


drop_columns = ["Unnamed: 0", "Unnamed: 0.1"]
dataset.drop(labels= drop_columns, axis=1, inplace=True)


# In[808]:


dataset.head()


# In[809]:


dataset.shape


# In[810]:


dataset.dtypes


# In[811]:


dataset['loan_status'].value_counts()


# In[812]:


dataset['education'].value_counts()


# In[813]:


dataset['Gender'].value_counts()


# In[814]:


# z = dataset.iloc[:, 0:1].values


# In[815]:


# Any text data needs to be converted into numbers that our model can use,
# We'll also fill any empty cells with 0:

# dataset = pd.get_dummies(dataset, columns=["education"])
# dataset.fillna(value=0.0, inplace=True)


# In[816]:


# Any text data needs to be converted into numbers that our model can use,
# We'll also fill any empty cells with 0:

# dataset = pd.get_dummies(dataset, columns=["Gender"])
# dataset.fillna(value=0.0, inplace=True)


# In[817]:


# Any text data needs to be converted into numbers that our model can use,
# We'll also fill any empty cells with 0:

# dataset = pd.get_dummies(dataset, columns=["loan_status"])
# dataset.fillna(value=0.0, inplace=True)


# In[818]:


dataset['due_date'] = pd.to_datetime(dataset['due_date'])
dataset['effective_date'] = pd.to_datetime(dataset['effective_date'])


# In[819]:


dataset.head()


# In[820]:


dataset.shape


# In[821]:


dataset.dtypes


# In[822]:


# notice: installing seaborn might takes a few minutes
# !conda install -c anaconda seaborn -y


# In[823]:


import seaborn as sns

bins = np.linspace(dataset.Principal.min(), dataset.Principal.max(), 10)
g = sns.FacetGrid(dataset, col="Gender", hue="loan_status", palette="Set1", col_wrap=2)
g.map(plt.hist, 'Principal', bins=bins, ec="k")

g.axes[-1].legend()
plt.show()


# In[824]:


bins = np.linspace(dataset.age.min(), dataset.age.max(), 10)
g = sns.FacetGrid(dataset, col="Gender", hue="loan_status", palette="Set1", col_wrap=2)
g.map(plt.hist, 'age', bins=bins, ec="k")

g.axes[-1].legend()
plt.show()


# In[825]:


dataset['dayofweek'] = dataset['effective_date'].dt.dayofweek
bins = np.linspace(dataset.dayofweek.min(), dataset.dayofweek.max(), 10)
g = sns.FacetGrid(dataset, col="Gender", hue="loan_status", palette="Set1", col_wrap=2)
g.map(plt.hist, 'dayofweek', bins=bins, ec="k")
g.axes[-1].legend()
plt.show()


# We see that people who get the loan at the end of the week dont pay it off, so lets use Feature binarization to set a threshold values less then day 4

# In[826]:


dataset['weekend'] = dataset['dayofweek'].apply(lambda x: 1 if (x>3)  else 0)
dataset.head()


# In[827]:


dataset.shape


# # Convert Categorical features to numerical values

# In[828]:


# Lets look at gender:
dataset.groupby(['Gender'])['loan_status'].value_counts(normalize=True)


# In[829]:


# Lets convert male to 0 and female to 1:
dataset['Gender'].replace(to_replace=['male','female'], value=[0,1],inplace=True)
dataset.head()


# In[830]:


dataset['loan_status'].replace(to_replace=['PAIDOFF','COLLECTION'], value=[0,1],inplace=True)
dataset.head()


# In[831]:


dataset['Gender'] = dataset['Gender'].astype('float')
dataset.head()


# In[832]:


dataset['Gender'] = dataset['Gender'].astype('int')
dataset.head()


# # One Hot Encoding

# How about education?

# In[833]:


dataset.groupby(['education'])['loan_status'].value_counts(normalize=True)

Feature before One Hot Encoding
# In[834]:


dataset[['loan_status','Principal','terms','age','Gender','education']].head()


# Use one hot encoding technique to conver categorical varables to binary variables and append them to the feature Data Frame

# In[835]:


Feature = dataset[['loan_status','Principal','terms','age','Gender']]
Feature = pd.concat([Feature,pd.get_dummies(dataset['education'])], axis=1)
Feature.drop(['Master or Above'], axis = 1,inplace=True)
Feature.head()


# In[836]:


dataset = Feature


# In[837]:


dataset.head()


# # Feature selection

# Lets defind feature sets, X:

# In[838]:


X = dataset
X[0:5]


# In[839]:


y = dataset['loan_status'].values
y[0:5]


# # Normalize Data

# Data Standardization give data zero mean and unit variance (technically should be done after train test split )

# X= preprocessing.StandardScaler().fit(X).transform(X)
# X[0:5]

# 

# 

# In[840]:


get_ipython().system('wget -O loan_test.csv https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/loan_test.csv')


# In[841]:


dataset_test = pd.read_csv('loan_test.csv')


# In[842]:


dataset_test.head()


# In[843]:


drop_columns = ["Unnamed: 0", "Unnamed: 0.1","due_date","effective_date"]
dataset_test.drop(labels= drop_columns, axis=1, inplace=True)


# In[844]:


dataset_test.shape


# In[845]:


# Lets convert male to 0 and female to 1:
dataset_test['Gender'].replace(to_replace=['male','female'], value=[0,1],inplace=True)
dataset_test.head()


# In[846]:


dataset_test['loan_status'].replace(to_replace=['PAIDOFF','COLLECTION'], value=[0,1],inplace=True)
dataset_test.head()


# In[847]:


Feature_test = dataset_test[['loan_status','Principal','terms','age','Gender']]
Feature_test = pd.concat([Feature_test,pd.get_dummies(dataset_test['education'])], axis=1)
Feature_test.drop(['Master or Above'], axis = 1,inplace=True)
Feature_test.head()


# In[848]:


dataset_test = Feature_test


# In[849]:


# dataset_test['education'].replace(to_replace=['Bechalor','Master or Above','High School or Below','college'], value=[0,1,2,3],inplace=True)
# dataset_test.head()


# In[850]:


#dataset_test['loan_status'] = dataset_test['loan_status'].astype('float')
dataset_test.head()


# In[851]:


dataset_test.shape


# In[852]:


X_train=dataset.iloc[:,1:8]
y_train=dataset.iloc[:,0:1]


# In[853]:


X_test = dataset_test.iloc[:,1:8]


# # K Nearest Neighbor(KNN)

# In[854]:


# Train Test Split

# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20,random_state=4)


# In[855]:


# Preprocessing

# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()
# scaler.fit(X_train)

# X_train = scaler.transform(X_train)
# X_test = scaler.transform(X_test)


# In[856]:


# from sklearn.neighbors import KNeighborsClassifier
# classifier = KNeighborsClassifier(n_neighbors = 5)
# classifier.fit(X_train, y_train)


# In[857]:


# y_pred = classifier.predict(X_test)


# In[858]:


# from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
# print(confusion_matrix(y_test, y_pred))
# print(classification_report(y_test, y_pred))
# print(accuracy_score(y_test, y_pred))


# In[859]:


# error = []

## Calculating error for K values between 1 and 40
# for i in range(1, 40):
#    knn = KNeighborsClassifier(n_neighbors=i)
#    knn.fit(X_train, y_train)
#    pred_i = knn.predict(X_test)
#    error.append(np.mean(pred_i != y_test))


# In[860]:


# plt.figure(figsize=(12, 6))
# plt.plot(range(1, 40), error, color='red', linestyle='dashed', marker='o',
#        markerfacecolor='blue', markersize=10)
# plt.title('Error Rate K Value')
# plt.xlabel('K Value')
# plt.ylabel('Mean Error')


# In[861]:


from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 7)
classifier.fit(X_train, y_train)


# In[862]:


y_pred = classifier.predict(X_test)


# In[863]:


# from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
# print(confusion_matrix(y_test, y_pred))
# print(classification_report(y_test, y_pred))
# print(accuracy_score(y_test, y_pred))


# In[864]:


# from sklearn.metrics import jaccard_similarity_score, f1_score, log_loss
# print(jaccard_similarity_score(y_test, y_pred))
# print(f1_score(y_test, y_pred))
# print(log_loss(y_test, y_pred)


# In[865]:


from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss


# # Decision Tree

# The X and y are the arrays required before the split, the test_size represents the ratio of the testing dataset, and the random_state ensures that we obtain the same splits.

# In[866]:


# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10)


# Inside of the classifier, specify criterion="entropy" so we can see the information gain of each node.

# In[867]:


from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion="entropy", max_depth = 2)
classifier # it shows the default parameters


# Next, we will fit the data with the training feature matrix X_trainset and training response vector y_trainset

# In[868]:


classifier.fit(X_train,y_train)


# # Prediction

# Let's make some predictions on the testing dataset and store it into a variable called y_pred

# In[869]:


y_pred = classifier.predict(X_test)


# # Evaluation

# Next, let's import metrics from sklearn and check the accuracy of our model.

# In[870]:


from sklearn import metrics
import matplotlib.pyplot as plt
# print("DecisionTrees's Accuracy: ", metrics.accuracy_score(y_test, y_pred))


# In[871]:


from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss


# 

# # Support Vector Machine

# In[872]:


##  the model_selection library of the Scikit-Learn library contains the train_test_split
## method that allows us to seamlessly divide data into training and test sets.

# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20,random_state = 10)


# In[873]:


# Training the Algorithm

from sklearn.svm import SVC   # Support Vector Classifier
classifier = SVC(kernel='linear')#This class takes one parameter,which is the kernel type
classifier.fit(X_train, y_train)


# In[874]:


y_pred = classifier.predict(X_test)


# In[875]:


# from sklearn.metrics import classification_report, confusion_matrix,accuracy_score
# print(confusion_matrix(y_test,y_pred))
# print(classification_report(y_test,y_pred))
# print(accuracy_score(y_test, y_pred))


# In[876]:


# from sklearn.metrics import jaccard_similarity_score, f1_score, log_loss
# print(jaccard_similarity_score(y_test, y_pred))
# print(f1_score(y_test, y_pred))
# print(log_loss(y_test, y_pred)


# In[877]:


from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss


# 

# # Kernel SVM

#  Polynomial Kernel

# In[878]:


# Polynomial Kernel
from sklearn.svm import SVC   # Support Vector Classifier
classifier = SVC(kernel ='poly', degree= 8)
classifier.fit(X_train, y_train)


# In[879]:


y_pred = classifier.predict(X_test)


# In[880]:


## Evaluating the Algorithm
# from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
# print(confusion_matrix(y_test, y_pred))
# print(classification_report(y_test, y_pred))
# print(accuracy_score(y_test, y_pred))


# In[881]:


# from sklearn.metrics import jaccard_similarity_score, f1_score, log_loss
# print(jaccard_similarity_score(y_test, y_pred))
# print(f1_score(y_test, y_pred))
# print(log_loss(y_test, y_pred)


# In[882]:


from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss


# 

# In[883]:


# Gaussian Kernel
from sklearn.svm import SVC
classifier = SVC(kernel='rbf')
classifier.fit(X_train, y_train)


# In[884]:


y_pred = classifier.predict(X_test)


# In[885]:


## Evaluating the Algorithm
# from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
# print(confusion_matrix(y_test, y_pred))
# print(classification_report(y_test, y_pred))
# print(accuracy_score(y_test, y_pred))


# In[886]:


# from sklearn.metrics import jaccard_similarity_score, f1_score, log_loss
# print(jaccard_similarity_score(y_test, y_pred))
# print(f1_score(y_test, y_pred))
# print(log_loss(y_test, y_pred)


# In[887]:


from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss


# 

# In[888]:


# Sigmoid Kernel
from sklearn.svm import SVC
classifier = SVC(kernel='sigmoid')
classifier.fit(X_train, y_train)


# In[889]:


y_pred = classifier.predict(X_test)


# In[890]:


## Evaluating the Algorithm
# from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
# print(confusion_matrix(y_test, y_pred))
# print(classification_report(y_test, y_pred))
# print(accuracy_score(y_test, y_pred))


# In[891]:


# from sklearn.metrics import jaccard_similarity_score, f1_score, log_loss
# print(jaccard_similarity_score(y_test, y_pred))
# print(f1_score(y_test, y_pred))
# print(log_loss(y_test, y_pred)


# In[892]:


from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss


# 

# # Logistic Regression

# In[893]:


# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=10)


# In[894]:


from sklearn.linear_model import LogisticRegression
LR = LogisticRegression(C=0.01, solver='liblinear')
LR.fit(X_train,y_train)


# In[895]:


y_pred = LR.predict(X_test)


# In[896]:


## Evaluating the Algorithm
# from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
# print(confusion_matrix(y_test, y_pred))
# print(classification_report(y_test, y_pred))
# print(accuracy_score(y_test, y_pred))


# In[897]:


# from sklearn.metrics import jaccard_similarity_score, f1_score, log_loss
# print(jaccard_similarity_score(y_test, y_pred))
# print(f1_score(y_test, y_pred))
# print(log_loss(y_test, y_pred)


# In[898]:


from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss


# 

# # Model Evaluation using Test set

# ### Load Test set for evaluation

# In[900]:


from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


# url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

# Assign colum names to the dataset
# names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']

# Read dataset to pandas dataframe
# dataset = pd.read_csv(url, names=names)

