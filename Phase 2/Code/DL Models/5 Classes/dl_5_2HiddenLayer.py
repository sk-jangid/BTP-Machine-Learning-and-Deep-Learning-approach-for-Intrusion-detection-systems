import numpy as np
from sklearn.metrics import confusion_matrix
import pandas as pd  
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from sklearn.model_selection import train_test_split  
from keras.layers import Dense
import time
from sklearn.metrics import confusion_matrix, classification_report



data_set= pd.read_csv('/content/drive/MyDrive/BTP/kddcup.data.csv') 
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

 
for i in range(len(y)):
  y[i] = attack_class[y[i]]

encoder = LabelEncoder()
encoder.fit(y)
encoded_Y = encoder.transform(y)
dummy_y = np_utils.to_categorical(encoded_Y)

x_train, x_test, y_train, y_test= train_test_split(x, dummy_y, test_size= 0.30, random_state=0)




model = Sequential()
model.add(Dense(12, input_dim=41, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(5, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

y_train = np.asarray(y_train).astype('float32')
x_train = np.asarray(x_train).astype('float32')
print(y_train[:10])
time_start = time.time()
model.fit(x_train, y_train, epochs=10, batch_size=10)
time_end = time.time()
print("Training time: ", time_end - time_start)

y_test = np.asarray(y_test).astype('float32')
x_test = np.asarray(x_test).astype('float32')

_, accuracy = model.evaluate(x_test, y_test)
print("Accuracy: ", accuracy)


time_start = time.time()
y_pred = model.predict(x_test)
time_end = time.time()
print("Testing time: ", time_end - time_start)

y_pred = np.argmax(y_pred, axis=1)
y_test = np.argmax(y_test, axis=1)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))