# This code to perform feature slection on a leave-one-user-out data set using SFF greedy algo 
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
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import make_scorer
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs



#intial veariables 
winSize=sys.argv[1]
testName=winSize+'sec_1sSlid_DS_lpf_all_4C_vR'+'_EarNeck_noMM_fft_autoCorFull0025'

classes =[ 'eating' , 'silent' , 'talking' , 'walking']#['other','walking'] #

mainClass='eating'

nTrees=100
projectDir='/media/mgoel1/86646EC1646EB399/Statistical_Datasets/'+testName+'/'
subjects = ['April2Sub1','April2Sub3', 'April2Sub4', 'March24Sub1','March24Sub3','March24Sub4',\
'March25Sub' , 'March28Sub1', 'March28Sub3', 'March28Sub4']



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
	if (tp+fp)==0:
		prec=0
	else:
		prec=100*tp/(tp+fp)
	if (tp+fp)==0:
		recall=0
	else:
		recall=100*tp/(tp+fn)
	if (prec+recall)==0:
		f1=0
	else:
		f1=2*(prec*recall)/(prec+recall)
	return acc,prec,recall,cm,f1

def my_custom_loss_func(ground_truth, predictions):
	#print "ground_truth=",len(ground_truth)
	#print ground_truth.shape, predictions.shape
	acc,prec,recall,cm,f1=analysis(ground_truth, predictions,mainClass)
	#print "f1= ",f1
	return f1

#load data

def loadDataOnce():
	allData=[]
	allLabel=[]
	groups=[]
	for i in range(len(subjects)):
		
		print "load ", subjects[i]
		#load training and testing data
		

		testDataFileName=projectDir+subjects[i]+'_testData.txt'
		testDataFile=open(testDataFileName,'r')
		testData=map(str.split,testDataFile)


		testData, tstLabels,ntestFile = zip(*[(s[2:], [s[0]],[s[1]]) for s in testData])
		testData=[[float(y) for y in x] for x in testData]
		testLabels=[]
		[[testLabels.append(y) for y in x] for x in tstLabels]
		print len(testData)
		allData=allData+testData
		allLabel=allLabel+testLabels
		groups=groups+[i+1]*len(testData)

	return allData,allLabel,groups

def genFeatuerName(fList):
	fList=list(fList)
	featureNames=[]
	featuresName=['var','rms','zc','zcv','ent','np','mxp','fpz','lp','hp','psd','mxA','mxF','DC']
	axisName=['Ex','Ey','Ez','Nx','Ny','Nz']
	featureCode=[]
	for x in axisName:
    		for f in featuresName:
        		featureCode.append(x+'_'+f)

	for i in fList:
   		 featureNames.append(featureCode[i])
	return featureNames

####### Featuer selection process using SFFS ######
resultsFileName=projectDir+'SFFS_FeatureSelectionResults.txt'
resultsFile=open(resultsFileName,'w')
#generate a score function that use F1 score for eating detection
score = make_scorer(my_custom_loss_func, greater_is_better=True)

#load data
allData,allLabel,groups=loadDataOnce()
nFeatures=len(allData[0])
X = np.array(allData)
y = np.array(allLabel)
#Generate splits 
logo = LeaveOneGroupOut()
sp=logo.split(X, y, groups=groups)
split=[]

for train_index, test_index in sp:
	#print("TRAIN:", train_index, "TEST:", test_index)
	#X_train, X_test = X[train_index], X[test_index]
	#y_train, y_test = y[train_index], y[test_index]
	#print(train_index.shape,test_index.shape)
	#print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
	split.append((train_index, test_index))

#estimator 
rfc=RandomForestClassifier(n_estimators=nTrees)

#feature selection
sffs = SFS(rfc,k_features=(1,nFeatures),forward=True,floating=True,verbose=2,scoring=score,cv=split,n_jobs=-1)
sffs = sffs.fit(X,y)
# print results
print('\nSequential Floating Forward Selection:')
print(sffs.k_feature_idx_)
print genFeatuerName(sffs.k_feature_idx_)
print('CV Score:')
print(sffs.k_score_)
print("Deataled results")
print(sffs.subsets_)
print ("full list of featuers:")
f= genFeatuerName(range(nFeatures))
for i in f:
    print f.index(i), i


############print to file #################

resultsFile.write('\nSequential Floating Forward Selection:\n')
resultsFile.write(str(sffs.k_feature_idx_)+"\n")
resultsFile.write(str(genFeatuerName(sffs.k_feature_idx_))+"\n")
resultsFile.write('\nCV Score:\n')
resultsFile.write(str(sffs.k_score_)+"\n")
resultsFile.write("\nDeataled results\n")
resultsFile.write(str(sffs.subsets_)+"\n")
resultsFile.write("\nfull list of featuers:\n")
f= genFeatuerName(range(nFeatures))
for i in f:
    resultsFile.write(str(f.index(i))+','+str(i))
    resultsFile.write('\n')
resultsFile.close()

#Plot results
plot_sfs(sffs.get_metric_dict(), kind='std_dev');
#plt.ylim([0.8, 1])
plt.title('Sequential Forward Selection (w. StdDev)')
plt.grid()
plt.show()
