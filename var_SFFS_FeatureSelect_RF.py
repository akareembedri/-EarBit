
# This code to perform feature slection on a leave-one-out data set 
#!/usr/bin/python
import sys
from sklearn.ensemble import RandomForestClassifier 
from sklearn.externals import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_selection import SelectFromModel
from statistics import mean,variance,stdev
import os
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.metrics import make_scorer 

#intial veariables 
winSize=sys.argv[1]
testName=winSize+'sec_1sSlid_DS_lpf_all_4C_vR'+'_EarNeck_noMM_fft_autoCorFull0025'

classes =[ 'eating' , 'silent' , 'talking' , 'walking']#['other','walking'] #
importance=[]
mainClass='eating'
CM=[[0 for col in range(len(classes))] for row in range(len(classes))]
nTrees=100
projectDir='/media/mgoel1/86646EC1646EB399/Statistical_Datasets/'+testName+'/'
subjects = ['April2Sub1','April2Sub3', 'April2Sub4', 'March24Sub1','March24Sub3','March24Sub4',\
'March25Sub' , 'March28Sub1', 'March28Sub3', 'March28Sub4']
nFeatuers=14*6
FeatuerLeft=[True]*nFeatuers

#exteact one column from a list
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
#analize results and return prec,recall, acc and confusion matrix
def analysis(testLabels,resultsValues,mainClass):
	tp=0
	tn=0
	fp=0
	fn=0
	cm=[[0 for col in range(len(classes))] for row in range(len(classes))]
	for l in range(len(testLabels)):
		if testLabels[l]==resultsValues[l] and resultsValues[l]==mainClass:
			tp+=1
		elif testLabels[l]!=resultsValues[l] and testLabels[l]==mainClass:
			fn+=1
		elif resultsValues[l]!=mainClass and testLabels[l]!=mainClass:
			tn+=1
		elif testLabels[l]!=resultsValues[l] and resultsValues[l]==mainClass:
			fp+=1
		

		cm[classes.index(testLabels[l])][classes.index(resultsValues[l])]+=1
		
	total=tp+tn+fp+fn
	acc=100*(tp+tn)/total
	prec=100*tp/(tp+fp)
	recall=100*tp/(tp+fn)
	return acc,prec,recall,cm

def loo_eating_score(gt,pred):
	#testData=gt[0]
	#trainData=gt[1]
	#testLabel=pred[0]
	#trainLabel=pred[1]
	print "data shape",len(gt), len(gt[0])
	print "prediction shape",len(pred),len(pred[0])
	return 0.11111


#load data
allTrainData=[]
allTestData=[]
allTrainLabel=[]
allTestLabel=[]
def loadData():
	for i in range(len(subjects)):
		
		print "load ", subjects[i]
		#load training and testing data
		trainDataFileName=projectDir+subjects[i]+'_trainData.txt'
		trainDataFile=open(trainDataFileName,'r')
		trainData=map(str.split,trainDataFile)

		testDataFileName=projectDir+subjects[i]+'_testData.txt'
		testDataFile=open(testDataFileName,'r')
		testData=map(str.split,testDataFile)

		trainData, trnLabels,ntrainFile = zip(*[(s[2:], [s[0]],[s[1]]) for s in trainData])
		testData, tstLabels,ntestFile = zip(*[(s[2:], [s[0]],[s[1]]) for s in testData])
		testData=[[float(y) for y in x] for x in testData]
		trainData=[[float(y) for y in x] for x in trainData]
		trainLabels=[]
		[[trainLabels.append(y) for y in x] for x in trnLabels]
		testLabels=[]
		[[testLabels.append(y) for y in x] for x in tstLabels]

		allTrainData.append(trainData)
		allTestData.append(testData)
		allTrainLabel.append(trainLabels)
		allTestLabel.append(testLabels)
	return allTrainData,allTestData,allTrainLabel,allTestLabel

AccMean=[]
PrecMean=[]
RecallMean=[]
ImportanceMean=[]

AccVar=[]
PrecVar=[]
RecallVar=[]

Importanceittr=[]
featureList=range(nFeatuers)


resultsFileName=projectDir+'FeatureSelectionResults.txt'
resultsFile=open(resultsFileName,'w')
allTrainData,allTestData,allTrainLabel,allTestLabel=loadData()
featureLeft=nFeatuers
trainData=[]
trainLabels=[]
testData=[]
testLabels=[]
count=0
oldnFeatures=0
loo_score=make_socrer(loo_eating_score, greater_is_better=True)
while(featureLeft>1 and featureLeft != oldnFeatures  ):
	count+=1
	oldnFeatures=featureLeft
	ACC=[]
	PREC=[]
	RECALL=[]
	IMPORT=[]
	for i in range(len(subjects)):
		resultsFile.write("=================================================================================\n")
        	resultsFile.write(subjects[i]+"\n")
		print subjects[i]
		#dfine dataset
		if (featureLeft==nFeatuers):
			trainData.append(allTrainData[i])
			trainLabels.append(allTrainLabel[i])
			testData.append(allTestData[i])
			testLabels.append(allTestLabel[i])
		else:
			trainData[i]=[]
			testData[i]=[]
			trainData[i]=trainData_new[i]
			testData[i]=testData_new[i]
			
		# define the RF prameters, train and test
	
		rfc=RandomForestClassifier(n_estimators=nTrees)
		#feature selection
		sffs = SFS(rfc,k_features=(1,3),forward=True,floating=True,verbose=2,scoring=loo_score,cv=0,n_jobs=-1)
		sffs = sffs.fit(np.array(trainData[i]),np.array(trainLabels[i]))
		#plot_sfs(sffs.get_metric_dict(), kind='std_err');
		'''rfc.fit(trainData[i],trainLabels[i])
		score=rfc.score(testData[i],testLabels[i])
		# detailed results
		resultsValues=rfc.predict(testData[i])
		# detailed results
		acc,prec,recall,cm=analysis(testLabels[i],resultsValues,mainClass)
		importance= rfc.feature_importances_  
		ACC.append(acc)
		PREC.append(prec)
		RECALL.append(recall)
		IMPORT.append(importance)
		############print to file #################
		#resultsFile.write("acc, pre, recall\n")
		#resultsFile.write(subjects[i]+"\n")
		#resultsFile.write("acc, pre, recall\n")
		#resultsFile.write(subjects[i]+"\n")
		#############################
		'''
		print sffs.subsets_
		print('\nSequential Floating Forward Selection:')
		print(sffs.k_feature_idx_)
		print('CV Score:')
		print(sffs.k_score_)
		#model = SelectFromModel(rfc, prefit=True, threshold=0.005)#, threshold=0.02
		#trainData_temp = sffs.transform(np.array(trainData[i]))


'''
		#newFeatureIndex=model.get_support(indices=False)
		#newFeatureIndex=maxFLength(newFeatureIndex,Oldind)
		print "==============================================================================="
		
		FeatuerLeft=[a and b for a,b in zip(FeatuerLeft, newFeatureIndex)]
		
		
		
	print "[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[["
	#save acc,prec,recall metrics for this itteration 
	AccMean.append(mean(ACC))
	PrecMean.append(mean(PREC))
	RecallMean.append(mean(RECALL))
        for i in range(featureLeft):
		ImportanceMean.append(mean(column(IMPORT,i)))

	AccVar.append(stdev(ACC))
	PrecVar.append(stdev(PREC))
	RecallVar.append(stdev(RECALL))
	#print ImportanceMean
	x=0
	Importanceittr.append([0]*nFeatuers)
	for i in range(nFeatuers):
		#print x
		#print i,featureList[x]
		if featureList[x]==i:
			Importanceittr[count-1][i]=ImportanceMean[x]
			if(x<len(featureList)-1):
				x+=1
		else:
			Importanceittr[count-1][i]=0
	#	ImportanceVar.append(stdev(column(IMPORT,i)))
	ImportanceMean=[]
	print Importanceittr
	#Generate next itteration dataset
	#index in new list
	newFeature=[]
	newFList=[]
	for i in range(len(FeatuerLeft)):
		if(FeatuerLeft[i]):
			newFeature.append(i)
			newFList.append(featureList[i]) 
	featureList=newFList
	#index in original list
	
	print featureLeft
	print FeatuerLeft
	#print newFeature
	print newFList
	featureLeft=len(newFeature)
	trainData_new=[]
	testData_new=[]
	for j in range(len(subjects)):
		trainData_new.append(transform(trainData[j],newFeature))
		testData_new.append(transform(testData[j],newFeature))
	



plt.figure(1)
plt.errorbar(range(count),AccMean, yerr=AccVar, marker='o')
plt.errorbar(range(count),PrecMean, yerr=PrecVar, marker='o')
plt.errorbar(range(count),RecallMean, yerr=RecallVar, marker='o')	
plt.legend(['accuracy', 'precision','recall'], loc='upper right')
axes = plt.gca()
axes.set_xlim([-1,count+1])
#axes.set_ylim([-0.1,1.2])
plt.show()	

plt.figure(2)
print ImportanceMean
for i in range(nFeatuers):
	plt.errorbar(range(count),column(Importanceittr, i), yerr=0.001, marker='o')

axes = plt.gca()
axes.set_xlim([-1,count+1])
#axes.set_ylim([-0.1,1.2])
plt.show()	

'''
	
	
	
