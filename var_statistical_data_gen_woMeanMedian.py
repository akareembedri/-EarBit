#This code generates statistical data sets for both RF and SVM
#!/usr/bin/python
import sys
import os
from statistics import mean, variance,median
from math import sqrt
import pandas as pd
import scipy as sc
from scipy.stats import entropy
import numpy as np
from math import log

winSize=sys.argv[1]
projectDir='datasets/awer_home/'
subjects = ['April2Sub1','April2Sub3', 'April2Sub4', 'March24Sub1','March24Sub3','March24Sub4',\
'March25Sub' , 'March28Sub1', 'March28Sub3', 'March28Sub4']
types  =[ 'eating' , 'silent' , 'talking' , 'walking']#'drinking' ['other','walking'] # 
dataSetName=winSize+'sec_1sSlid_DS_lpf_all_4C_vR'

genDataSet='datasets/Statistical_Datasets/'+dataSetName+'_EarNeck7fEntropy/'
if not os.path.exists(genDataSet):
		os.makedirs(genDataSet)
senCol=[1,2,4,6,8,10]
nFeatures=5

def cEntropy(data):
	p_data= pd.Series(data).value_counts()/len(data) # calculates the probabilities
	entropy=sc.stats.entropy(p_data)  # input probabilities to get the entropy 
	return entropy
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
	return variance(y)
def cMean(data):
	return mean(data)
def cVar(data):
	return variance(data)
def cRMS(data):
	ms = 0
	for y in range(len(data)):
    		ms = ms + pow(data[y],2)
	ms = ms / len(data)
	rms = sqrt(ms)
	return rms
def cMedian(data):
	return median(data)

#go through each  user dependent data set and generate features
bins = [-10000000000] + range(-50, 51, 10) + [10000000000]
for i in range(len(subjects)):
	print subjects[i]
	dirPath=projectDir+subjects[i]+'/ALL/'+dataSetName+'/UserDependentTraining/'
	testFile=open(genDataSet+subjects[i]+'_testData.txt','w')
	data=[0]*(nFeatures*len(senCol))
	
	for typ in types:
		path=dirPath+typ
		#print path
		if not os.path.exists(path):
			os.makedirs(path)
		listFiles = os.listdir(path)
		#print listFiles
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
				#data[senCol.index(col)*nFeatures+4]=cMean(column(framData,col))
				#data[senCol.index(col)*nFeatures+5]=cMedian(column(framData,col))
				#data[senCol.index(col)*nFeatures+6]=cEntropy(column(framData,col))
				data[senCol.index(col)*nFeatures+4]=cEntropyBins(column(framData,col), bins)
			#add to testing dataset
			testFile.write("%s" % (typ)+ " ")
			testFile.write("%s" %(int(files.split('_')[1]))+ " ")
			for n in range(nFeatures*len(senCol)):
				testFile.write('%.5f'%(data[n])+ " ")
			testFile.write("\n")
			#add to training datasets
			for j in range(len(subjects)):
				if j!=i:
					trainFile=open(genDataSet+subjects[j]+'_trainData.txt','a')
					trainFile.write("%s" % (typ)+ " ")
					trainFile.write("%s" %(int(files.split('_')[1]))+ " ")

					for n in range(nFeatures*len(senCol)):
						trainFile.write('%.5f'%(data[n])+ " ")
					trainFile.write("\n")
					trainFile.close()

	testFile.close()



