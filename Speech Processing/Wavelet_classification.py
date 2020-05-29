#############

# IMPORTANT#
# location needs to be changed/verified in the line 87,88,125 
# for  proper working of the code

#############

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os 
import pywt
import sklearn
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import glob
import scipy.io.wavfile as wav
import math
import csv


def normal(signal):
    l = []
    m = np.amax(signal)
    for i in np.nditer(signal):
        l.append((i/m)*5)
    signal = np.asarray(l)
    return signal
def energy_calculation(signal):
    l = len(signal)
    ans = 0 
    for i in range(len(signal)):
        ans = ans + (signal[i]*signal[i])
    return ans
    #return math.log(ans,1)
def calculation(signal,band):
    l = len(signal)
    p = l//4
    #calculation for first frame
    signal1 = signal[0:p]
    wp1 = pywt.WaveletPacket(data=signal1,wavelet='db5',mode='symmetric')
    #calculation for second frame
    signal2 = signal[p:2*p]
    wp2 = pywt.WaveletPacket(data=signal2,wavelet='db5',mode='symmetric')
    #calculation for third frame
    signal3 = signal[2*p:3*p]
    wp3 = pywt.WaveletPacket(data=signal3,wavelet='db5',mode='symmetric')
    #calculation for fourth frame
    signal4 = signal[3*p:4*p]
    wp4 = pywt.WaveletPacket(data=signal4,wavelet='db5',mode='symmetric')
    
    #path to reach particular frequency band
    path = ''
    if(band==1):
        path = 'aaaa'
    elif(band == 2):
        path = 'aaad'
    elif(band == 3):
        path = 'aadd'
    elif(band == 4):
        path = 'aada'
    elif(band == 5):
        path = 'add'
    elif(band == 6):
        path = 'ada'
    elif(band == 7):
        path = 'dd'
    elif(band == 8):
        path = 'da'
    #separate energy calculation for each frame  
    #later mean energy of all four frame with each having different weight factor, is taken as coefficient
    x1 = energy_calculation(np.array(wp1[path].data))
    x2 = energy_calculation(np.array(wp2[path].data))
    x3 = energy_calculation(np.array(wp3[path].data))
    x4 = energy_calculation(np.array(wp4[path].data))
    return ((x1+3*x2+3*x3+x4)/8)

def perform(signal):
    ans = []
    signal = normal(signal)
    for i in range(1,9):
        ans.append(calculation(signal,i))
    return np.array(ans)

#home loc to code
code_location = ""
os.chdir(code_location + "/HINDI_DATA/05_test/train")

audio_data_train = {}
lable_train = []
files = os.listdir()
for file in files:
    if file.startswith("0_"):
        lable_train.append(0)
    elif file.startswith("1_"):
        lable_train.append(1)
    elif file.startswith("2_"):
        lable_train.append(2)
    elif file.startswith("3_"):
        lable_train.append(3)
    elif file.startswith("4_"):
        lable_train.append(4)
    elif file.startswith("5_"):
        lable_train.append(5)
    elif file.startswith("6_"):
        lable_train.append(6)
    elif file.startswith("7_"):
        lable_train.append(7)
    elif file.startswith("8_"):
        lable_train.append(8)
    elif file.startswith("9_"):
        lable_train.append(9)
    
    
    rate,data = wav.read(file)
    cd = perform(data)
    
    audio_data_train[file] = cd
    

for i in audio_data_train:
    print(i,audio_data_train[i])

os.chdir(code_location + "/HINDI_DATA/05_test/test")


audio_data_test = {}
lable_test = []
files = os.listdir()
for file in files:
    if file.startswith("0_"):
        lable_test.append(0)
    elif file.startswith("1_"):
        lable_test.append(1)
    elif file.startswith("2_"):
        lable_test.append(2)
    elif file.startswith("3_"):
        lable_test.append(3)
    elif file.startswith("4_"):
        lable_test.append(4)
    elif file.startswith("5_"):
        lable_test.append(5)
    elif file.startswith("6_"):
        lable_test.append(6)
    elif file.startswith("7_"):
        lable_test.append(7)
    elif file.startswith("8_"):
        lable_test.append(8)
    elif file.startswith("9_"):
        lable_test.append(9)
    
    
    rate,data = wav.read(file)
    cd = perform(data)
    
    audio_data_test[file] = cd
    

for i in audio_data_test:
    print(i,audio_data_test[i])


#generating a  list where each element of list is a feature vector 

list_train = []
for i in audio_data_train:
    list_train.append(audio_data_train[i])
    
list_test = []
for i in audio_data_test:
    list_test.append(audio_data_test[i])
# conversion into numpy array
test = np.asarray(list_test)
train = np.asarray(list_train)


clf = sklearn.svm.SVC(C=0.5,kernel='linear')
clf.fit(list_train,lable_train)
y_pred = clf.predict(list_test)
print("Accuracy: %.2f%%" % (sklearn.metrics.accuracy_score(lable_test,y_pred)*100))

