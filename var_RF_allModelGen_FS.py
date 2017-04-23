 
# This code train and test Leave-One user -Out random forest models and save them in sapreate file
#!/usr/bin/python
import sys
from sklearn.ensemble import RandomForestClassifier 
from sklearn.externals import joblib
from statistics import mean
import numpy as np
import os

winSize=sys.argv[1]


def column(matrix, i):
    return [row[i] for row in matrix]
#resahpe an array with specific indecies 
def transform(Data,indices):
	newData=np.array([])
	for i in indices:
		temp=column(Data, i)
		temp=np.asarray(temp)
		if (i==indices[0]):
			newData=temp
		else:
			newData=np.column_stack((newData,temp))
	return newData	

testName=winSize+'sec_1sSlid_DS_lpf_all_4C_vR'+'_EarNeck_noMM_fft_autoCorFull0025'


nTrees=100
projectDir='/media/mgoel1/86646EC1646EB399/Statistical_Datasets/'+testName+'/'
subjects = 'April2Sub1'
sfIndx=[4, 5, 7, 11, 12, 14, 15, 17, 21, 24, 26, 27, 28, 32, 36, 37, 38, 44, 45, 46, 49, 50, 54, 61, 62, 67, 68, 72, 74, 75, 76, 77, 82, 83]#selected featuers indcies 

print len(sfIndx) 

modelName='All_'+str(len(sfIndx))+'BestFeatures'



	
#load training and testing data
trainDataFileName=projectDir+subjects+'_trainData.txt'
trainDataFile=open(trainDataFileName,'r')
trainData=map(str.split,trainDataFile)

testDataFileName=projectDir+subjects+'_testData.txt'
testDataFile=open(testDataFileName,'r')
testData=map(str.split,testDataFile)
	
trainData=trainData+testData #have all data in training

trainData, trnLabels,ntrainFile = zip(*[(s[2:], [s[0]],[s[1]]) for s in trainData])
	
trainData=[[float(y) for y in x] for x in trainData]
trainLabels=[]
[[trainLabels.append(y) for y in x] for x in trnLabels]
	
trainData=transform(trainData,sfIndx)#select featuers 
print len(trainData[0])
# define the RF prameters and train 
	
	
rfc=RandomForestClassifier(n_estimators=nTrees)
rfc.fit(trainData,trainLabels)
	
print rfc.feature_importances_

#save models
if not os.path.exists(projectDir+'/models'):
	os.makedirs(projectDir+'/models')
joblib.dump(rfc,projectDir+'/models/'+modelName+'.pk1')
