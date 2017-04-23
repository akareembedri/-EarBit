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
from detect_peaks import detect_peaks
from scipy import signal
winSize=sys.argv[1]
projectDir='datasets/awer_home/'
subjects = ['April2Sub1','April2Sub3', 'April2Sub4', 'March24Sub1','March24Sub3','March24Sub4','March25Sub' , 'March28Sub1', 'March28Sub3', 'March28Sub4']
#user=['kareem_1a','kareem_2a','kareem_b','malcolm_a','malcolm_b','richard_a','richard_b','sarah_a','sarah_b','lu_b','lin_a','lin_b','angela_a','cheng_a','cheng_b','jz_a','sharly_a']

types  =['testDataset']#[ 'eating' , 'silent' , 'talking' , 'walking']#'drinking' ['other','walking'] # 
dataSetName=winSize+'sec_1sSlid_DS_lpf_all_4C_vR'
testName='_EarNeck_noMM_fft_autoCorFull0025'
#projectDir='/root/soundBite/datasets/Statistical_Datasets/'+dataSetName+testName+'/'
genDataSet='/media/mgoel1/86646EC1646EB399/Statistical_Datasets/'+dataSetName+testName+'/'


if not os.path.exists(genDataSet):
		os.makedirs(genDataSet)
senCol=[1,2,4,6,8,10]
nFeatures=14
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
def autocorr(x):
    result = np.correlate(x, x, mode='full')
    return result[result.size/2:]
def autocorrFeatures(x):
	thrsh=0.025
	data=autocorr(x)
	normData=[float(n/float(max(x))) for n in x ]
	peaks = detect_peaks(normData, show=False)
        nPleaks=len(peaks)
	maxPeak=max([normData[p] for p in peaks])
	#geteing first zero crossing
	zcdata=[a*b for a,b in zip(normData[:-1],normData[1:])]
	for z in range(len(zcdata)):
		if zcdata[z]<0:
			firsZC=z
			break
	for p in peaks:
		if p>firsZC:
			firstPeakAfterZC=p
			break
	hpeaks=0
	lpeaks=0
	for i in range(len(peaks)):
		if i==0:
			if normData[peaks[i]]>normData[peaks[i+1]]+thrsh:
				hpeaks+=1
			elif normData[peaks[i]]<normData[peaks[i+1]]-thrsh:
				lpeaks+=1
		elif i==len(peaks)-1:
			if normData[peaks[i]]>normData[peaks[i-1]]+thrsh:
				hpeaks+=1
			elif normData[peaks[i]]<normData[peaks[i-1]]-thrsh:
				lpeaks+=1
		else:
			if normData[peaks[i]]>normData[peaks[i-1]]+thrsh and normData[peaks[i]]>normData[peaks[i+1]]+thrsh:
				hpeaks+=1
			elif normData[peaks[i]]<normData[peaks[i-1]]-thrsh and normData[peaks[i]]<normData[peaks[i+1]]-thrsh:
				lpeaks+=1
			
	
	return nPleaks,maxPeak, firstPeakAfterZC, lpeaks, hpeaks

def fftFeatures(data):
	#fft
	Fs = 10.0;  # sampling rate
	Ts = 1.0/Fs; # sampling interval
	n = len(data) # length of the signal
	t = np.arange(0,n/Fs,Ts) # time vector
	
	k = np.arange(n)
	T = n/Fs
	frq = k/T # two sides frequency range
	frq = frq[range(n/2)] # one side frequency range

	Y = np.fft.fft(data)/n # fft computing and normalization
	Y = Y[range(n/2)]
	#plot
	#removed
	#featuer computation
	maxAmp=max(abs(Y))
	maxFreq=frq[np.argmax(abs(Y))]
	psd= 1.0*sum(abs(Y))/n
	real=Y.real
	pwr=np.power(real, 2)
	DC=1.0*sum(pwr)/n

	return psd,maxAmp,maxFreq,DC

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
				#data[senCol.index(col)*nFeatures+4]=cMean(column(framData,col))
				#data[senCol.index(col)*nFeatures+5]=cMedian(column(framData,col))
				#data[senCol.index(col)*nFeatures+6]=cEntropy(column(framData,col))
				data[senCol.index(col)*nFeatures+4]=cEntropyBins(column(framData,col), bins)
				nPleaks,maxPeak,firstPeakAfterZC, lpeaks, hpeaks = autocorrFeatures(column(framData,col))
				data[senCol.index(col)*nFeatures+5]=nPleaks
				data[senCol.index(col)*nFeatures+6]=maxPeak
				data[senCol.index(col)*nFeatures+7]=firstPeakAfterZC
				data[senCol.index(col)*nFeatures+8]=lpeaks
				data[senCol.index(col)*nFeatures+9]=hpeaks
				psd,maxAmp,maxFreq,DC=fftFeatures(column(framData,col))
				data[senCol.index(col)*nFeatures+10]=psd
				data[senCol.index(col)*nFeatures+11]=maxAmp
				data[senCol.index(col)*nFeatures+12]=maxFreq
				data[senCol.index(col)*nFeatures+13]=DC
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
	
	
