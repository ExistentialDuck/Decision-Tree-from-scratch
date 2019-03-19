
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


import warnings
warnings.filterwarnings('ignore')


# In[2]:


from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.tree import DecisionTreeClassifier


df = pd.read_csv('/home/stephen/Classes/Machine Learning/Hanna-hw1/Problem 2.csv')
B = df.drop(['A'], axis=1)
X2 = B.drop(['Outcome'], axis=1)
y = df['Outcome']
print(B)


# In[3]:


Decider = DecisionTreeClassifier(criterion = "entropy")
# Train the model using the training sets and check score
Decider.fit(X2, y)
#Predict Output
Decider_predicted= Decider.predict(X2)

Decider_score = round(Decider.score(X2, y) * 100, 2)
Decider_score_test = round(Decider.score(X2, y) * 100, 2)
#Equation coefficient and Intercept
print('Deision Tree Training Score: \n', Decider_score)
print('Decision Tree Test Score: \n', Decider_score_test)
print('Accuracy: \n', accuracy_score(y,Decider_predicted))
print('Confusion Matrix: \n', confusion_matrix(y,Decider_predicted))
print('Classification Report: \n', classification_report(y,Decider_predicted))

sns.heatmap(confusion_matrix(y,Decider_predicted),annot=True,fmt="d")


# In[4]:


from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus
from sklearn.feature_extraction import DictVectorizer
dot_data = StringIO()

export_graphviz(Decider, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True, feature_names = list(X2))

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())


# Since there is the same number of instances for values + and - for both a1 and a2, we can just average the entropy values for their first nodes, which is 1 in this case. Alternatively, we can use the shannon entropy formula and do  __-P(a1=1)*log(1/P(a1=1)) - P(a2=1)*log(1/P(a2=1))__ which also comes out to 1. Thus, the entropy of this collection of training samples, with respect to the target classification, is 1.

# Using the Shannon entropy formula listed above, the entropy for a2 being + is 1, shown in the bottom right node. The entropy for a2 being - is 1 as well, shown in the bottom left node. The weighted entropy is 4/6 and 2/6, respectively, due to the sample size for each outcome relative to the overall sample size. __Summed, the entropy is 1, making the information gain 0 since IG = 1 - Entropy__
