#!/usr/bin/env python
# coding: utf-8

# In[3]:


# In this jupyter notebook we will be using scikit-learn to determine which of the two predictor columns that you selected in the last assignment most accurately predicts whether or not a mushroom is poisonous. 
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import numpy as np
import pandas as pd
pd.set_option('display.notebook_repr_html',True) 
pd.set_option('display.max_rows', None) 
pd.set_option('display.max_columns', None)

sns.set(style='whitegrid',font_scale=4)
sns.set_color_codes('pastel')

dfFungi = pd.read_csv("data/agaricus-lepiota.data.txt",                       header = None,                       usecols = [0, 3, 5,],                       names = ['poisonous', 'color', 'odor'])
#We are going to create a dictionary to hold the values for the replacement of the strings to integers. 
dictNewValues = {
    'e' : 0, \
    'p' : 1
}
#We wil be replacing the cell values with the values in the dictionary
dfFungi.replace({'poisonous': dictNewValues}, inplace=True)
dfFungi.head(5)
#We are going to create a dictionary to hold the values for the replacement of the strings to integers. 
dictNewValues = {
    'n' : 0, \
    'b' : 1, \
    'c' : 2, \
    'g' : 3, \
    'r' : 4, \
    'p' : 5, \
    'u' : 6, \
    'e' : 7, \
    'w' : 8, \
    'y' : 9, \
}
#We wil be replacing the cell values with the values in the dictionary
dfFungi.replace({'color': dictNewValues}, inplace=True)
dfFungi.head(5)
#We are going to create a dictionary to hold the values for the replacement of the strings to integers. 
dictNewValues = {
    'a' : 0, \
    'l' : 1, \
    'c' : 2, \
    'y' : 3, \
    'f' : 4, \
    'm' : 5, \
    'n' : 6, \
    'p' : 7, \
    's' : 8
}
#We wil be replacing the cell values with the values in the dictionary
dfFungi.replace({'odor': dictNewValues}, inplace=True)
dfFungi.head(5)
dfFungi.shape
#We are going to create 3 series with the first 8000 observations
sPoisonous = pd.Series(dfFungi.poisonous[:7999])
sColor = pd.Series(dfFungi.color[:7999])
sOdor = pd.Series(dfFungi.odor[:7999])
#We are going to assign those observations to a new dataframe containing the TRAINING data
dfTraining = pd.DataFrame({'color':sColor, 'odor':sOdor, 'poisonous':sPoisonous})

sPoisonous = pd.Series(dfFungi.poisonous[8000:])
sColor = pd.Series(dfFungi.color[8000:])
sOdor = pd.Series(dfFungi.odor[8000:])
#We are going to assign those observations to a new dataframe containing the TESTING data
dfTests = pd.DataFrame({'color':sColor, 'odor':sOdor, 'poisonous':sPoisonous})
print(dfTraining.dtypes)
print(dfTests.dtypes)
#We will be creating  3 different datasets to test against.
knnColor = KNeighborsClassifier(n_neighbors= 5)
knnOdor = KNeighborsClassifier(n_neighbors= 5)
knnAll = KNeighborsClassifier(n_neighbors= 5)
#We will be fitting the color estimator with the values from the dtraining dataset
knnColor.fit(dfTraining[['color']].values,              dfTraining[['poisonous']].values.ravel())           
#We will be fitting the odor estimator with the values from the dtraining dataset
knnOdor.fit(dfTraining[['odor']].values,             dfTraining[['poisonous']].values.ravel())
#We will be fitting the all estimator with the values from the dtraining dataset
knnAll.fit(dfTraining[['color','odor']].values,            dfTraining[['poisonous']].values.ravel())
#We are going to test the color estimator against the testing dataset and store the results in the testing dataframe.
dfTests['preColor'] = pd.Series(knnColor.predict(dfTests[['color']].values), dfTests.index)print(metrics.accuracy_score(                              dfTests[['poisonous']].values.ravel(),                              dfTests[['preColor']].values))
#We are going to test the odor estimator against the testing dataset and store the results in the testing dataframe.
dfTests['preOdor'] = pd.Series(knnOdor.predict(dfTests[['odor']].values), dfTests.index)
print(metrics.accuracy_score(                              dfTests[['poisonous']].values.ravel(),                              dfTests[['preOdor']].values))
#We are going to test the all estimator against the testing dataset and store the results in the testing dataframe.
dfTests['preAll'] = pd.Series(knnAll.predict(dfTests[['color','odor']].values), dfTests.index)
print(metrics.accuracy_score(                              dfTests[['poisonous']].values.ravel(),                              dfTests[['preAll']].values))
#We will be creating a scatterplot that shows the relationship between odor and poisonous
sns.lmplot('odor', 'poisonous', data=dfTests, size=30, aspect=1.294, scatter_kws={"s": 1200}).savefig('Output/Test Data - Observations Only.pdf')
plt.title("Chart 1: Test Data : Observations Only")
#We will be creating a scatterplot that shows the relationship between odor and poisonous
sns.lmplot('odor', 'preOdor', data=dfTests, size=30, aspect=1.294, scatter_kws={"s": 1200}).savefig('Output/Test Data - Prediction vs Observation.pdf')
plt.title("Chart 2: Test Data : Prediction vs Observation")
# We will be testing the odor estimator against the TRAINING dataset and store the results in the TRAINING dataframe.
dfTraining['preOdor'] = pd.Series(knnOdor.predict(dfTraining[['odor']].values), dfTraining.index)
# We will be creating a scatterplot that shows the relationship between odor and poisonous
sns.lmplot('odor', 'poisonous', data=dfTraining, size=30, aspect=1.294, scatter_kws={"s": 1200}).savefig('Output/Training Data - Observations Only.pdf')
plt.title("Chart 3: Training Data : Observations Only")
# We will be creating a scatterplot that shows the relationship between odor and the predicted poisonous with our training data
sns.lmplot('odor', 'preOdor', data=dfTraining, size=30, aspect=1.294, scatter_kws={"s": 1200}).savefig('Output/Training Data - Prediction vs Observation.pdf')
plt.title("Chart 4: Training Data : Prediction vs Observation")
# We will be printing the accuracy score of the odor TRAINING estimator
print(metrics.accuracy_score(                              dfTraining[['poisonous']].values.ravel(),                              dfTraining[['preOdor']].values))

