#This code loads seqential data frame by frame-generate featuers and save them in text file - load models and test data on it- save results and confidance
#!/usr/bin/python
import sys
import os
from statistics import mean, variance, median
from math import sqrt, log
from sklearn.externals import joblib

import pandas as pd
import scipy as sc
from scipy.stats import entropy
import numpy as np

winSize=sys.argv[1]
projectDir='datasets/awer_home/'
subjects = ['April2Sub1','April2Sub3', 'April2Sub4', 'March24Sub1','March24Sub3','March24Sub4','March25Sub' , 'March28Sub1', 'March28Sub3', 'March28Sub4']
#user=['kareem_1a','kareem_2a','kareem_b','malcolm_a','malcolm_b','richard_a','richard_b','sarah_a','sarah_b','lu_b','lin_a','lin_b','angela_a','cheng_a','cheng_b','jz_a','sharly_a']

types  =['testDataset']#[ 'eating' , 'silent' , 'talking' , 'walking']#'drinking' ['other','walking'] # 
dataSetName=winSize+'sec_1sSlid_DS_lpf_all_4C_vR'
testName='_EarNeck7fEntropy'
#projectDir='/root/soundBite/datasets/Statistical_Datasets/'+dataSetName+testName+'/'
genDataSet='datasets/Statistical_Datasets/'+dataSetName+testName+'/'


if not os.path.exists(genDataSet):
		os.makedirs(genDataSet)
senCol=[1,2,4,6,8,10]
nFeatures=7
def cEntropyBins(data, bins):
	hist, bin_edges = np.histogram(data, bins=bins, density=True)
	entropy = -1*sum([p_i*log(p_i) for p_i in hist if not p_i==0])
	return entropy

def column(matrix, i):
    return [row[i] for row in matrix]

def cZeroCrossing(data):
	x=[a*b for a,b in zip(data[:-1],data[1:])]
	c=0
	for z in x:
		if z<0:
			c+=1
	return c
def cZeroCrossingVar(data):
	res=[]	
	x=[a*b for a,b in zip(data[:-1],data[1:])]
	c=0
	for z in x:
		if z<0:
			res.append(x.index(z))
	y=[b-a for a,b in zip(res[:-1],res[1:])]
	if len(y)>1:
		return variance(y)
	else:
		return 0
def cMean(data):
	return mean(data)
def cMedian(data):
	return median(data)
def cVar(data):
	return variance(data)
def cRMS(data):
	ms = 0
	for y in range(len(data)):
    		ms = ms + pow(data[y],2)
	ms = ms / len(data)
	rms = sqrt(ms)
	return rms

#loads seqential data frame by frame-generate featuers and save them in text file
bins = [-10000000000] + range(-50, 51, 10) + [10000000000]
for i in range(len(subjects)):
	print subjects[i]
	dirPath=projectDir+subjects[i]+'/ALL/'+dataSetName+'/UserDependentTraining/'
	testFile=open(genDataSet+subjects[i]+'_SeqTestData.txt','w')
	predictFile=open(genDataSet+subjects[i]+'_predictResults.txt','w')
	data=[0]*(nFeatures*len(senCol))
	
	for typ in types:
		path=dirPath+typ
		#print path
		listFiles = sorted(os.listdir(path))
		#print listFiles
		fram=[[0]*(nFeatures*len(senCol))]*len(listFiles)
		print len(fram)
		for files in listFiles:
			#print files
			dataFile=open(path+'/'+files,'r')
			framData=map(str.split,dataFile)
			framData=[[float(y) for y in x] for x in framData]
			for col in senCol:
				#print column(framData,col)
				
				data[senCol.index(col)*nFeatures]=cVar(column(framData,col))
				data[senCol.index(col)*nFeatures+1]=cRMS(column(framData,col))
				data[senCol.index(col)*nFeatures+2]=cZeroCrossing(column(framData,col))
				data[senCol.index(col)*nFeatures+3]=cZeroCrossingVar(column(framData,col))
				data[senCol.index(col)*nFeatures+4]=cMean(column(framData,col))
				data[senCol.index(col)*nFeatures+5]=cMedian(column(framData,col))
				data[senCol.index(col)*nFeatures+6]=cEntropyBins(column(framData,col), bins)
			index=int(files.split('_')[1])
			fram[index]=data
			
			data=[0]*(nFeatures*len(senCol))
		#add to testing dataset
	for j in fram:
		testFile.write("%s" % (typ)+ " ")
		testFile.write("%s" %(fram.index(j))+ " ")
		for n in range(nFeatures*len(senCol)):
			
			testFile.write('%.5f'%(j[n])+ " ")
		testFile.write("\n")

	testFile.close()
	
	#print fram[:10]
#load models and test data on it
	clf = joblib.load(genDataSet+'models/'+subjects[i]+'.pk1')
	predict=clf.predict(fram)
	results=clf.predict_proba(fram)
	print clf.classes_
	print results[:10]
	predictFile.write("class %s\n" % (clf.classes_)+ " ")
	for j in range(len(fram)):
		predictFile.write("%s" % (predict[j])+ " ")

		for n in range(len(results[j])):
			predictFile.write('%.5f'%(results[j][n])+ " ")
		predictFile.write("\n")
	predictFile.close()

