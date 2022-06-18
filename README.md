# Name: Satyam Mishra

# Task 6: Prediction using Decision Tree Algorithm

# GRIP@TheSparkFoundation - Data Science and Business Analytics - JUNE2022

# Objective

● Create the Decision Tree Classifier and visualize it graphically.

● The purpose is if we feed any new data to this classifier, It would
  be able to predict the right class Accordingaly.

# Import the all required libraries.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.tree import plot_tree
import seaborn as sns

# Reading the data

dataset=pd.read_csv("C:\csvfile\iris.csv")
dataset.head(5)

sns.pairplot(data=dataset,hue="Species")
plt.show()

x=dataset.iloc[:,1:5]

y=dataset.iloc[:,5:]

from sklearn.preprocessing import LabelEncoder
lmn=LabelEncoder()
y=lmn.fit_transform(y)

# Now let,s defining the Decision Tree Algorithm

from sklearn.tree import DecisionTreeClassifier
dtc=DecisionTreeClassifier()

dtc.fit(x,y)
print("Decision Tree Classifer Created")

# Let,s visualize ths Decision Tree to understand it better.

plt.figure(figsize=(20,20))
d_t=plot_tree(decision_tree=dtc,feature_names=dataset.columns,class_names=["setosa","virginica","versicolor"],filled=True)
plt.show()

# **You can now feed any new/test data to this classifer and it would be able to predict the right class accordingly.**
