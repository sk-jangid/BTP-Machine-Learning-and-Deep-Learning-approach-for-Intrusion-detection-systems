# -*- coding: utf-8 -*-
"""ML_Algorithms.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1thqyO-mNEmBu8MYnv71oRYS5pM2UAc-M

Mount Drive
"""

from google.colab import drive
from google.colab import files
print("Getting Data from Google drive")
drive.mount('/content/drive/', force_remount ="True")
#text_en = open("/content/drive/MyDrive/BTP/kddcup.data.csv","r").read()

"""Get *data* and Preprocessing"""

import numpy as nm  
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as mtp  
import pandas as pd  
data_set= pd.read_csv('/content/drive/MyDrive/BTP/kddcup.data.csv') 
#data_Set = data_set.dropna('columns')
x= data_set.iloc[:, 0:41].values 
y= data_set.iloc[:, 41].values  
d = {}
# Counting data poihnts of each class
for val in y:
  if val in d:
    d[val] = d[val]+1
  else:
    d[val] = 1

print(d)

from sklearn.preprocessing import LabelEncoder  
label_encoder_x1= LabelEncoder()  
label_encoder_x2= LabelEncoder() 
label_encoder_x3= LabelEncoder()  
    
x[:, 1]= label_encoder_x1.fit_transform(x[:, 1])  
x[:, 2]= label_encoder_x2.fit_transform(x[:, 2])  
x[:, 3]= label_encoder_x3.fit_transform(x[:, 3]) 

attack_class = {'normal.': 'Normal', 'back.': 'DOS', 'smurf.':'DOS', 'neptune.':'DOS', 'pod.': 'DOS', 'teardrop.':'DOS', 'land.': 'DOS',
                'multihop.':'R2L', 'ftp_write.': 'R2L', 'guess_passwd.': 'R2L', 'imap.': 'R2L', 'phf.': 'R2L', 'spy.':'R2L', 'warezmaster.':'R2L', 'warezclient.':'R2L',
                'rootkit.':'U2R', 'loadmodule.':'U2R', 'buffer_overflow.': 'U2R', 'perl.': 'U2R',
                'portsweep.':'PROBE', 'satan.':'PROBE', 'ipsweep.': 'PROBE', 'nmap.':'PROBE'}

from sklearn.model_selection import train_test_split  
x_train, x_test, y_train, y_test= train_test_split(x, y, test_size= 0.10, random_state=0) 



for i in range(len(y_train)):
  y_train[i] = attack_class[y_train[i]]

for i in range(len(y_test)):
  y_test[i] = attack_class[y_test[i]]


y_train2 = y_train.copy()
y_test2 = y_test.copy()
for i in range(len(y_train2)):
  if y_train2[i] == 'Normal':
    y_train2[i] = 'Normal'
  else:
    y_train2[i] = 'Intrusion'

for i in range(len(y_test2)):
  if y_test2[i] == 'Normal':
    y_test2[i] = 'Normal'
  else:
    y_test2[i] = 'Intrusion'

print(len(y))

import time
from sklearn import metrics

"""## **K-Means**"""

from sklearn.cluster import KMeans
kmeans1 = KMeans(n_clusters = 5)
time_start = time.time()
kmeans1.fit(x_train)
time_end = time.time()
print("Training time: ", time_end - time_start)

time_start = time.time()
y_pred = kmeans1.predict(x_test)
time_end = time.time()
print("Testing time: ", time_end - time_start)
print(y_pred)

for i in range(5):
  print(list(y_pred).count(i))

print(nm.unique(y_pred))

from sklearn.cluster import KMeans

kmeans2 = KMeans(n_clusters = 2)
time_start = time.time()
kmeans2.fit(x_train)
time_end = time.time()

print("Training time: ", time_end - time_start)
time_start = time.time()
y_pred2 = kmeans2.predict(x_train)
time_end = time.time()
print("Testing time: ", time_end - time_start)
print(y_pred2)

for i in range(2):
  print(list(y_pred2).count(i))


print(nm.unique(y_pred2))

"""## **Random Forest**"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

classifier1 = RandomForestClassifier(n_estimators = 50)
time_start = time.time()
classifier1.fit(x_train, y_train)
time_end = time.time()
print("Training time: ", time_end - time_start)

time_start = time.time()
y_pred = classifier1.predict(x_test)
time_end = time.time()
print("Testing time: ", time_end - time_start)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
result1 = classification_report(y_test, y_pred, zero_division = 0)
print(result1)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

classifier2 = RandomForestClassifier(n_estimators = 50)
time_start = time.time()
classifier2.fit(x_train, y_train2)
time_end = time.time()
print("Training time: ", time_end - time_start)

time_start = time.time()
y_pred2 = classifier2.predict(x_test)
time_end = time.time()
print("Testing time: ", time_end - time_start)
print("Accuracy:",metrics.accuracy_score(y_test2, y_pred2))
print(confusion_matrix(y_test2, y_pred2))
result2 = classification_report(y_test2, y_pred2)
print(result2)

"""## **Naive- Bayes Classifier**"""

from sklearn.naive_bayes import GaussianNB

gnb1 = GaussianNB()
time_start = time.time()
gnb1.fit(x_train, y_train)
time_end   = time.time()
print("Training time: ", time_end - time_start)

time_start = time.time()
y_pred = gnb1.predict(x_test)
time_end = time.time()
print("Testing time: ", time_end - time_start)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
result1 = classification_report(y_test, y_pred, zero_division =  0)
print(confusion_matrix(y_test, y_pred))
print(result1)

from sklearn.metrics import confusion_matrix
print(nm.unique(y_pred))
print(confusion_matrix(y_test, y_pred))

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report


gnb2 = GaussianNB()
time_start = time.time()
gnb2.fit(x_train, y_train2)
time_end = time.time()
print("Training time: ", time_end - time_start)

time_start = time.time()
y_pred2 = gnb2.predict(x_test)
time_end = time.time()
print("Testing time: ", time_end - time_start)
print("Accuracy:",metrics.accuracy_score(y_test2, y_pred2))
result2 = classification_report(y_test2, y_pred2)
print(confusion_matrix(y_test2, y_pred2))
print(result2)

"""## **Decision Tree**"""

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

clfd = DecisionTreeClassifier(criterion ="entropy", max_depth = 4)
time_start = time.time()
clfd.fit(x_train, y_train)
time_end = time.time()
print("Training time: ", time_end - time_start)

time_start = time.time()
y_pred = clfd.predict(x_test)
time_end = time.time()
print("Testing time: ", time_end- time_start)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
result = classification_report(y_test, y_pred, zero_division = 0)
print(confusion_matrix(y_test, y_pred))
print(result)

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

clfd = DecisionTreeClassifier(criterion ="entropy", max_depth = 4)
time_start = time.time()
clfd.fit(x_train, y_train2)
time_end = time.time()
print("Training time: ", time_end - time_start)

time_start = time.time()
y_pred2 = clfd.predict(x_test)
time_end = time.time()
print("Testing time: ", time_end - time_start)
print("Accuracy:",metrics.accuracy_score(y_test2, y_pred2))
result = classification_report(y_test2, y_pred2, zero_division = 0)
print(confusion_matrix(y_test2, y_pred2))
print(result)