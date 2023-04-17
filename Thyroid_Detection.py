# Importing essential libraries
import numpy as np
import pandas as pd
import pickle
import os
import random
import matplotlib.pyplot as plt
import seaborn as sns
from time import time
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, log_loss, accuracy_score
from sklearn.metrics import mean_squared_error


# Loading the dataset
data = pd.read_csv(r"C:\Users\vedav\Desktop\major_thyroid\thyroidDF_new.csv")

# Assigning required variable in new variable.
data_1 = data[['age','sex','on_thyroxine','on_antithyroid_meds','pregnant','thyroid_surgery','tumor','TSH','T3','TT4','T4U','FTI','target']]

data_1['sex'] = data_1['sex'].map({'F' : 0, 'M' : 1})
data_1[['on_thyroxine','on_antithyroid_meds','thyroid_surgery','pregnant','tumor']] = data_1[['on_thyroxine','on_antithyroid_meds',
                                                                                              'thyroid_surgery','pregnant','tumor']].replace({'f' : 0, 't' : 1})

# re-mapping target vaues to diagnostic groups
diagnoses = {'-': 'negative',
             'A': 'hyperthyroid', 
             'B': 'hyperthyroid', 
             'C': 'hyperthyroid', 
             'D': 'hyperthyroid',
             'E': 'Primary hypothyroid', 
             'F': 'Primary hypothyroid', 
             'G': 'Compensated hypothyroid', 
             'H': 'Secondary hypothyroid'}
data_1['target'] = data_1['target'].map(diagnoses) # re-mapping

# dropping observations with 'target' null after re-mapping
data_1.dropna(subset=['target'], inplace=True)
data_1.dropna(subset=['age'], inplace=True)
print(data_1.isnull().sum())

data_1['sex'].fillna(data_1['sex'].median(), inplace=True)
data_1['TT4'].fillna(data_1['TT4'].median(), inplace=True)
data_1['T4U'].fillna(data_1['T4U'].median(), inplace=True)
data_1['FTI'].fillna(data_1['FTI'].median(), inplace=True)
data_1['TSH'].fillna(data_1['TSH'].median(), inplace=True)
data_1['T3'].fillna(data_1['T3'].median(), inplace=True)


data_1.to_csv("Cleaned_thyroid_dataset.csv")

print(data_1.isnull().sum())

data_1['target'] = data_1['target'].map({'negative' : 0, 'hyperthyroid' : 1,'Primary hypothyroid':2,'Compensated hypothyroid':3,'Secondary hypothyroid':4})
print(data_1)

X = data_1.drop(columns = ['target'], axis = 1)
Y = data_1['target']
print(X.shape)
print(Y.shape)

# Model Building
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y,test_size = 0.2, random_state=1, stratify=Y)
print(X_train.shape,X_test.shape)
print(Y_train.shape,Y_test.shape)

# Creating Gradient Boosting Model..........
from sklearn.ensemble import GradientBoostingClassifier
# Define Gradient Boosting Classifier with hyperparameters
gbc=GradientBoostingClassifier(n_estimators=500,learning_rate=0.05,random_state=100,max_features=5 )
# Fit train data to GBC
gbc.fit(X_train,Y_train)
Y_pred = gbc.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_pred)
print ("Confusion Matrix : \n", cm)

print ("Accuracy : ", accuracy_score(Y_test, Y_pred))
data_1.info()

# Creating a pickle file for the classifier
filename = 'thyroid_detection_gbc_model.pkl'
pickle.dump(gbc, open(filename, 'wb'))


