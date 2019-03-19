
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
eps = np.finfo(float).eps
from math import log, e


# In[3]:


dffirst = pd.read_csv('/home/stephen/Classes/Machine Learning/Hanna-hw1/optdigits.tra.csv', header = None)
df = dffirst.head(25)
dftest = pd.read_csv('/home/stephen/Classes/Machine Learning/Hanna-hw1/optdigits.tes.csv', header = None)


# In[5]:


df10 = dffirst.head(380)
df20 = dffirst.head(760)
df50 = dffirst.head(1900)
df80 = dffirst.head(3040)
df100 = dffirst.head(3823)

dffirst = dffirst.sample(frac=1).reset_index(drop=True)
df101 = dffirst.head(380)
df201 = dffirst.head(760)
df501 = dffirst.head(1900)
df801 = dffirst.head(3040)
df1001 = dffirst.head(3823)

dffirst = dffirst.sample(frac=1).reset_index(drop=True)
df102 = dffirst.head(380)
df202 = dffirst.head(760)
df502 = dffirst.head(1900)
df802 = dffirst.head(3040)
df1002 = dffirst.head(3823)


# In[ ]:


def unique_vals(rows, col):
    return df[df.columns[col]].unique()


# In[ ]:


def class_counts(rows):
    counts = rows.iloc[:,-1].value_counts()
    return counts


# In[ ]:


class Question:
    def __init__(self, column, value):
        self.column = column
        self.value = value

    def match(self, example):
        val = example[self.column]
        return val >= self.value

    def __repr__(self):
        condition = ">="
        return "Is column %s %s %s?" % (self.column, condition, str(self.value))


# In[ ]:


def partition(rows, question):
    true_rows, false_rows = [], []
    for i in range(len(rows)):
        if question.match(rows.loc[i]):
            true_rows.append(rows.loc[i])
        else:
            false_rows.append(rows.loc[i])
    return true_rows, false_rows


# In[ ]:


def entropy(rows):
    counts = class_counts(rows)
    unique = rows.iloc[:,-1].unique()
    chaos = 0
    for i in unique:
        prob_of_lbl = counts[i] / (len(rows))
        chaos += - prob_of_lbl*log(prob_of_lbl, 2)
    return chaos


# In[ ]:


def info_gain(left, right, current_uncertainty):
    p = float(len(left)) / (len(left) + len(right))
    return current_uncertainty - p * entropy(left) - (1 - p) * entropy(right)


# In[ ]:


def find_best_split(rows):
    best_gain = 0  
    best_question = None  
    current_uncertainty = entropy(rows)
    n_features = len(list(rows)) - 1

    for col in range(n_features):  

        values = rows.iloc[:,-1].unique() 
        depthCount = 0

        for val in values:  

            question = Question(col, val)

            true_rows, false_rows = partition(rows, question)
            true_rows = pd.DataFrame(true_rows)
            false_rows = pd.DataFrame(false_rows)
            depthCount += 1

            if len(true_rows) == 0 or len(false_rows) == 0 or depthCount == 15:
                continue

            gain = info_gain(true_rows, false_rows, current_uncertainty)

            if gain >= best_gain:
                best_gain, best_question = gain, question
    return best_gain, best_question


# In[ ]:


class Leaf:
    def __init__(self, rows):
        self.predictions = class_counts(rows)


# In[ ]:


class Decision_Node:
    def __init__(self,
                 question,
                 true_branch,
                 false_branch):
        self.question = question
        self.true_branch = true_branch
        self.false_branch = false_branch


# In[ ]:


def build_tree(rows):
    gain, question = find_best_split(rows)

    if gain == 0:
        return Leaf(rows)

    true_rows, false_rows = partition(rows, question)
    true_rows = pd.DataFrame(true_rows)
    false_rows = pd.DataFrame(false_rows)
    true_rows.index = range(len(true_rows))
    false_rows.index = range(len(false_rows))

    true_branch = build_tree(true_rows)

    false_branch = build_tree(false_rows)

    return Decision_Node(question, true_branch, false_branch)


# In[ ]:


def print_tree(node, spacing=""):
    """World's most elegant tree printing function."""

    if isinstance(node, Leaf):
        print (spacing + "Predict", node.predictions)
        return

    print (spacing + str(node.question))

    print (spacing + '--> True:')
    print_tree(node.true_branch, spacing + "  ")

    print (spacing + '--> False:')
    print_tree(node.false_branch, spacing + "  ")


# In[ ]:


def classify(row, node):
    """See the 'rules of recursion' above."""

    if isinstance(node, Leaf):
        return node.predictions

    if node.question.match(row):
        return classify(row, node.true_branch)
    else:
        return classify(row, node.false_branch)


# In[ ]:


def print_leaf(counts):
    """A nicer way to print the predictions at a leaf."""
    total = sum(counts) * 1.0
    probs = {}
    i = 0
    for lbl in counts.index[:]:
        probs[lbl] = str(int(counts[lbl] / total * 100)) + "%"
        i += 1
    return probs


# In[ ]:


my_tree = build_tree(df)
def accuracy(dftest):
    testcases = pd.DataFrame(0, index=np.arange(10), columns=['Test Cases'])
    true = pd.DataFrame(0, index=np.arange(10), columns=['True'])
    false = pd.DataFrame(0, index=np.arange(10), columns=['False'])
    error = pd.DataFrame(0, index=np.arange(10), columns=['Error'])
    for j in range(0,10):
        cut = dftest[dftest.iloc[:,-1] == j]
        total = len(cut)
        testcases.iloc[j,0] = total
        correct = 0
        for i in range(total):
            left = cut.iloc[i,-1]
            right = classify(cut.iloc[i,:], my_tree).index[:]
            if left == right:
                true.iloc[j,0] +=1
            else:
                false.iloc[j,0] +=1
            error.iloc[j,0] = false.iloc[j,0]/total
    frame = pd.concat([testcases, true, false], axis=1)
    return frame

Error = accuracy(dftest)['False'].sum()/accuracy(dftest)['Test Cases'].sum()
print(accuracy(dftest))
print("The error using the first 25 rows is " + str(Error))


# In[ ]:


my_tree = build_tree(df10)
def accuracy(dftest):
    testcases = pd.DataFrame(0, index=np.arange(10), columns=['Test Cases'])
    true = pd.DataFrame(0, index=np.arange(10), columns=['True'])
    false = pd.DataFrame(0, index=np.arange(10), columns=['False'])
    error = pd.DataFrame(0, index=np.arange(10), columns=['Error'])
    for j in range(0,10):
        cut = dftest[dftest.iloc[:,-1] == j]
        total = len(cut)
        testcases.iloc[j,0] = total
        correct = 0
        for i in range(total):
            left = cut.iloc[i,-1]
            right = classify(cut.iloc[i,:], my_tree).index[:]
            if left == right:
                true.iloc[j,0] +=1
            else:
                false.iloc[j,0] +=1
            error.iloc[j,0] = false.iloc[j,0]/total
    frame = pd.concat([testcases, true, false], axis=1)
    return frame

Error = accuracy(dftest)['False'].sum()/accuracy(dftest)['Test Cases'].sum()
print(accuracy(dftest))
print("The error using the first 10% of rows is " + str(Error))

averageError = Error

my_tree = build_tree(df101)
def accuracy(dftest):
    testcases = pd.DataFrame(0, index=np.arange(10), columns=['Test Cases'])
    true = pd.DataFrame(0, index=np.arange(10), columns=['True'])
    false = pd.DataFrame(0, index=np.arange(10), columns=['False'])
    error = pd.DataFrame(0, index=np.arange(10), columns=['Error'])
    for j in range(0,10):
        cut = dftest[dftest.iloc[:,-1] == j]
        total = len(cut)
        testcases.iloc[j,0] = total
        correct = 0
        for i in range(total):
            left = cut.iloc[i,-1]
            right = classify(cut.iloc[i,:], my_tree).index[:]
            if left == right:
                true.iloc[j,0] +=1
            else:
                false.iloc[j,0] +=1
            error.iloc[j,0] = false.iloc[j,0]/total
    frame = pd.concat([testcases, true, false], axis=1)
    return frame

Error = accuracy(dftest)['False'].sum()/accuracy(dftest)['Test Cases'].sum()
print(accuracy(dftest))
print("The error using the first 10% of rows is " + str(Error))

averageError += Error

my_tree = build_tree(df102)
def accuracy(dftest):
    testcases = pd.DataFrame(0, index=np.arange(10), columns=['Test Cases'])
    true = pd.DataFrame(0, index=np.arange(10), columns=['True'])
    false = pd.DataFrame(0, index=np.arange(10), columns=['False'])
    error = pd.DataFrame(0, index=np.arange(10), columns=['Error'])
    for j in range(0,10):
        cut = dftest[dftest.iloc[:,-1] == j]
        total = len(cut)
        testcases.iloc[j,0] = total
        correct = 0
        for i in range(total):
            left = cut.iloc[i,-1]
            right = classify(cut.iloc[i,:], my_tree).index[:]
            if left == right:
                true.iloc[j,0] +=1
            else:
                false.iloc[j,0] +=1
            error.iloc[j,0] = false.iloc[j,0]/total
    frame = pd.concat([testcases, true, false], axis=1)
    return frame

Error = accuracy(dftest)['False'].sum()/accuracy(dftest)['Test Cases'].sum()
print(accuracy(dftest))
print("The error using the first 10% of rows is " + str(Error))

averageError += Error
error10 = averageError/3


# In[ ]:


my_tree = build_tree(df20)
def accuracy(dftest):
    testcases = pd.DataFrame(0, index=np.arange(10), columns=['Test Cases'])
    true = pd.DataFrame(0, index=np.arange(10), columns=['True'])
    false = pd.DataFrame(0, index=np.arange(10), columns=['False'])
    error = pd.DataFrame(0, index=np.arange(10), columns=['Error'])
    for j in range(0,10):
        cut = dftest[dftest.iloc[:,-1] == j]
        total = len(cut)
        testcases.iloc[j,0] = total
        correct = 0
        for i in range(total):
            left = cut.iloc[i,-1]
            right = classify(cut.iloc[i,:], my_tree).index[:]
            if left == right:
                true.iloc[j,0] +=1
            else:
                false.iloc[j,0] +=1
            error.iloc[j,0] = false.iloc[j,0]/total
    frame = pd.concat([testcases, true, false], axis=1)
    return frame

Error = accuracy(dftest)['False'].sum()/accuracy(dftest)['Test Cases'].sum()
print(accuracy(dftest))
print("The error using the first 20% of rows is " + str(Error))

averageError = Error

my_tree = build_tree(df201)
def accuracy(dftest):
    testcases = pd.DataFrame(0, index=np.arange(10), columns=['Test Cases'])
    true = pd.DataFrame(0, index=np.arange(10), columns=['True'])
    false = pd.DataFrame(0, index=np.arange(10), columns=['False'])
    error = pd.DataFrame(0, index=np.arange(10), columns=['Error'])
    for j in range(0,10):
        cut = dftest[dftest.iloc[:,-1] == j]
        total = len(cut)
        testcases.iloc[j,0] = total
        correct = 0
        for i in range(total):
            left = cut.iloc[i,-1]
            right = classify(cut.iloc[i,:], my_tree).index[:]
            if left == right:
                true.iloc[j,0] +=1
            else:
                false.iloc[j,0] +=1
            error.iloc[j,0] = false.iloc[j,0]/total
    frame = pd.concat([testcases, true, false], axis=1)
    return frame

Error = accuracy(dftest)['False'].sum()/accuracy(dftest)['Test Cases'].sum()
print(accuracy(dftest))
print("The error using the first 20% of rows is " + str(Error))

averageError += Error

my_tree = build_tree(df202)
def accuracy(dftest):
    testcases = pd.DataFrame(0, index=np.arange(10), columns=['Test Cases'])
    true = pd.DataFrame(0, index=np.arange(10), columns=['True'])
    false = pd.DataFrame(0, index=np.arange(10), columns=['False'])
    error = pd.DataFrame(0, index=np.arange(10), columns=['Error'])
    for j in range(0,10):
        cut = dftest[dftest.iloc[:,-1] == j]
        total = len(cut)
        testcases.iloc[j,0] = total
        correct = 0
        for i in range(total):
            left = cut.iloc[i,-1]
            right = classify(cut.iloc[i,:], my_tree).index[:]
            if left == right:
                true.iloc[j,0] +=1
            else:
                false.iloc[j,0] +=1
            error.iloc[j,0] = false.iloc[j,0]/total
    frame = pd.concat([testcases, true, false], axis=1)
    return frame

Error = accuracy(dftest)['False'].sum()/accuracy(dftest)['Test Cases'].sum()
print(accuracy(dftest))
print("The error using the first 20% of rows is " + str(Error))

averageError += Error

error20 = averageError/3


# In[ ]:


my_tree = build_tree(df50)
def accuracy(dftest):
    testcases = pd.DataFrame(0, index=np.arange(10), columns=['Test Cases'])
    true = pd.DataFrame(0, index=np.arange(10), columns=['True'])
    false = pd.DataFrame(0, index=np.arange(10), columns=['False'])
    error = pd.DataFrame(0, index=np.arange(10), columns=['Error'])
    for j in range(0,10):
        cut = dftest[dftest.iloc[:,-1] == j]
        total = len(cut)
        testcases.iloc[j,0] = total
        correct = 0
        for i in range(total):
            left = cut.iloc[i,-1]
            right = classify(cut.iloc[i,:], my_tree).index[:]
            if left == right:
                true.iloc[j,0] +=1
            else:
                false.iloc[j,0] +=1
            error.iloc[j,0] = false.iloc[j,0]/total
    frame = pd.concat([testcases, true, false], axis=1)
    return frame

Error = accuracy(dftest)['False'].sum()/accuracy(dftest)['Test Cases'].sum()
print(accuracy(dftest))
print("The error using the first 50% of rows is " + str(Error))

averageError = Error

my_tree = build_tree(df501)
def accuracy(dftest):
    testcases = pd.DataFrame(0, index=np.arange(10), columns=['Test Cases'])
    true = pd.DataFrame(0, index=np.arange(10), columns=['True'])
    false = pd.DataFrame(0, index=np.arange(10), columns=['False'])
    error = pd.DataFrame(0, index=np.arange(10), columns=['Error'])
    for j in range(0,10):
        cut = dftest[dftest.iloc[:,-1] == j]
        total = len(cut)
        testcases.iloc[j,0] = total
        correct = 0
        for i in range(total):
            left = cut.iloc[i,-1]
            right = classify(cut.iloc[i,:], my_tree).index[:]
            if left == right:
                true.iloc[j,0] +=1
            else:
                false.iloc[j,0] +=1
            error.iloc[j,0] = false.iloc[j,0]/total
    frame = pd.concat([testcases, true, false], axis=1)
    return frame

Error = accuracy(dftest)['False'].sum()/accuracy(dftest)['Test Cases'].sum()
print(accuracy(dftest))
print("The error using the first 50% of rows is " + str(Error))

averageError += Error

my_tree = build_tree(df502)
def accuracy(dftest):
    testcases = pd.DataFrame(0, index=np.arange(10), columns=['Test Cases'])
    true = pd.DataFrame(0, index=np.arange(10), columns=['True'])
    false = pd.DataFrame(0, index=np.arange(10), columns=['False'])
    error = pd.DataFrame(0, index=np.arange(10), columns=['Error'])
    for j in range(0,10):
        cut = dftest[dftest.iloc[:,-1] == j]
        total = len(cut)
        testcases.iloc[j,0] = total
        correct = 0
        for i in range(total):
            left = cut.iloc[i,-1]
            right = classify(cut.iloc[i,:], my_tree).index[:]
            if left == right:
                true.iloc[j,0] +=1
            else:
                false.iloc[j,0] +=1
            error.iloc[j,0] = false.iloc[j,0]/total
    frame = pd.concat([testcases, true, false], axis=1)
    return frame

Error = accuracy(dftest)['False'].sum()/accuracy(dftest)['Test Cases'].sum()
print(accuracy(dftest))
print("The error using the first 50% of rows is " + str(Error))

averageError += Error
error50 = averageError/3


# In[ ]:


my_tree = build_tree(df80)
def accuracy(dftest):
    testcases = pd.DataFrame(0, index=np.arange(10), columns=['Test Cases'])
    true = pd.DataFrame(0, index=np.arange(10), columns=['True'])
    false = pd.DataFrame(0, index=np.arange(10), columns=['False'])
    error = pd.DataFrame(0, index=np.arange(10), columns=['Error'])
    for j in range(0,10):
        cut = dftest[dftest.iloc[:,-1] == j]
        total = len(cut)
        testcases.iloc[j,0] = total
        correct = 0
        for i in range(total):
            left = cut.iloc[i,-1]
            right = classify(cut.iloc[i,:], my_tree).index[:]
            if left == right:
                true.iloc[j,0] +=1
            else:
                false.iloc[j,0] +=1
            error.iloc[j,0] = false.iloc[j,0]/total
    frame = pd.concat([testcases, true, false], axis=1)
    return frame

Error = accuracy(dftest)['False'].sum()/accuracy(dftest)['Test Cases'].sum()
print(accuracy(dftest))
print("The error using the first 80% of rows is " + str(Error))

averageError = Error

my_tree = build_tree(df801)
def accuracy(dftest):
    testcases = pd.DataFrame(0, index=np.arange(10), columns=['Test Cases'])
    true = pd.DataFrame(0, index=np.arange(10), columns=['True'])
    false = pd.DataFrame(0, index=np.arange(10), columns=['False'])
    error = pd.DataFrame(0, index=np.arange(10), columns=['Error'])
    for j in range(0,10):
        cut = dftest[dftest.iloc[:,-1] == j]
        total = len(cut)
        testcases.iloc[j,0] = total
        correct = 0
        for i in range(total):
            left = cut.iloc[i,-1]
            right = classify(cut.iloc[i,:], my_tree).index[:]
            if left == right:
                true.iloc[j,0] +=1
            else:
                false.iloc[j,0] +=1
            error.iloc[j,0] = false.iloc[j,0]/total
    frame = pd.concat([testcases, true, false], axis=1)
    return frame

Error = accuracy(dftest)['False'].sum()/accuracy(dftest)['Test Cases'].sum()
print(accuracy(dftest))
print("The error using the first 80% of rows is " + str(Error))

averageError += Error

my_tree = build_tree(df802)
def accuracy(dftest):
    testcases = pd.DataFrame(0, index=np.arange(10), columns=['Test Cases'])
    true = pd.DataFrame(0, index=np.arange(10), columns=['True'])
    false = pd.DataFrame(0, index=np.arange(10), columns=['False'])
    error = pd.DataFrame(0, index=np.arange(10), columns=['Error'])
    for j in range(0,10):
        cut = dftest[dftest.iloc[:,-1] == j]
        total = len(cut)
        testcases.iloc[j,0] = total
        correct = 0
        for i in range(total):
            left = cut.iloc[i,-1]
            right = classify(cut.iloc[i,:], my_tree).index[:]
            if left == right:
                true.iloc[j,0] +=1
            else:
                false.iloc[j,0] +=1
            error.iloc[j,0] = false.iloc[j,0]/total
    frame = pd.concat([testcases, true, false], axis=1)
    return frame

Error = accuracy(dftest)['False'].sum()/accuracy(dftest)['Test Cases'].sum()
print(accuracy(dftest))
print("The error using the first 80% of rows is " + str(Error))

averageError += Error
error80 = averageError/3


# In[ ]:


my_tree = build_tree(df100)
def accuracy(dftest):
    testcases = pd.DataFrame(0, index=np.arange(10), columns=['Test Cases'])
    true = pd.DataFrame(0, index=np.arange(10), columns=['True'])
    false = pd.DataFrame(0, index=np.arange(10), columns=['False'])
    error = pd.DataFrame(0, index=np.arange(10), columns=['Error'])
    for j in range(0,10):
        cut = dftest[dftest.iloc[:,-1] == j]
        total = len(cut)
        testcases.iloc[j,0] = total
        correct = 0
        for i in range(total):
            left = cut.iloc[i,-1]
            right = classify(cut.iloc[i,:], my_tree).index[:]
            if left == right:
                true.iloc[j,0] +=1
            else:
                false.iloc[j,0] +=1
            error.iloc[j,0] = false.iloc[j,0]/total
    frame = pd.concat([testcases, true, false], axis=1)
    return frame

Error = accuracy(dftest)['False'].sum()/accuracy(dftest)['Test Cases'].sum()
print(accuracy(dftest))
print("The error using all rows is " + str(Error))
error100 = Error


# In[ ]:


X = [10,20,50,80,100]
Y = [error10,error20,error50,error80,error100]


# In[ ]:


ax = plt.axes()
ax.plot(X,Y)
ax.set(xlabel='% of training data', ylabel='% error',
       title='Learning Curve');

