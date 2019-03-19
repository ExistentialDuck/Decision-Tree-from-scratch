
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


import warnings
warnings.filterwarnings('ignore')


# In[2]:


df = pd.read_csv('/home/stephen/Classes/Machine Learning/Hanna-hw1/Problem 1a.csv')
df.head(8)


# In[3]:


from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

X = df.drop(['Outcome'], axis=1)
y = df['Outcome']


# In[4]:


Decider = DecisionTreeClassifier(criterion = "entropy")
# Train the model using the training sets and check score
Decider.fit(X, y)
#Predict Output
Decider_predicted= Decider.predict(X)

Decider_score = round(Decider.score(X, y) * 100, 2)
Decider_score_test = round(Decider.score(X, y) * 100, 2)
#Equation coefficient and Intercept
print('Deision Tree Training Score: \n', Decider_score)
print('Decision Tree Test Score: \n', Decider_score_test)
print('Accuracy: \n', accuracy_score(y,Decider_predicted))
print('Confusion Matrix: \n', confusion_matrix(y,Decider_predicted))
print('Classification Report: \n', classification_report(y,Decider_predicted))

sns.heatmap(confusion_matrix(y,Decider_predicted),annot=True,fmt="d")


# In[5]:


from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus
from sklearn.feature_extraction import DictVectorizer
dot_data = StringIO()

export_graphviz(Decider, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True, feature_names = list(X))

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())

