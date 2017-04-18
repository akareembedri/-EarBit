
# This code train and test Leave-One user -Out random forest models and save them in sapreate file
#!/usr/bin/python
import sys
from sklearn.ensemble import RandomForestClassifier 
from sklearn.externals import joblib
from statistics import mean
import os

winSize=sys.argv[1]
def column(matrix, i):
    return [row[i] for row in matrix]

	

testName=winSize+'sec_1sSlid_DS_lpf_all_4C_vR'+'_Ear7fEntropy'

classes =[ 'eating' , 'silent' , 'talking' , 'walking']#['other','walking'] #
importance=[]
mainClass='eating'
CM=[[0 for col in range(len(classes))] for row in range(len(classes))]
nTrees=100
projectDir='datasets/Statistical_Datasets/'+testName+'/'
subjects = ['April2Sub1','April2Sub3', 'April2Sub4', 'March24Sub1','March24Sub3','March24Sub4',\
'March25Sub' , 'March28Sub1', 'March28Sub3', 'March28Sub4']

Tp=0
Tn=0
Fp=0
Fn=0
results=[]
resultsFileName=projectDir+'results.txt'
resultsFile=open(resultsFileName,'w')

for i in range(len(subjects)):
	resultsFile.write("=================================================================================\n")
        resultsFile.write(subjects[i]+"\n")
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
	# define the RF prameters, train and test
	
	rfc=RandomForestClassifier(n_estimators=nTrees)
	rfc.fit(trainData,trainLabels)
	score=rfc.score(testData,testLabels)
	# detailed results
	resultsValues=rfc.predict(testData)
	#resultsprob=rfc.predict_log_proba(testData)
	#comparison with lables
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
		resultsFile.write(" label= %s" % testLabels[l] )
		resultsFile.write(" %s" % ntestFile[l] )
		resultsFile.write(" recognized as= %s\n" % resultsValues[l] )
		#print  testLabels[l],classes.index(testLabels[l]),resultsValues[l],classes.index(resultsValues[l])
		cm[classes.index(testLabels[l])][classes.index(resultsValues[l])]+=1
		
	total=tp+tn+fp+fn
	acc=100*(tp+tn)/total
	prec=100*tp/(tp+fp)
	recall=100*tp/(tp+fn)
        #update overall values 

	for s in cm:
		for d in s:
			CM[cm.index(s)][s.index(d)]=CM[cm.index(s)][s.index(d)]+ d
	#[[CM[cm.index(s)][s.index(d)]+ d for d in s]for s in cm]
	#print cm
	#print CM
	#print [[CM[cm.index(s)][s.index(d)]+ d for d in s]for s in cm]
	Tp=Tp+tp
	Tn=Tn+tn
	Fp=Fp+fp
	Fn=Fn+fn
	#save and show results
	resultsFile.write(" %s\n" % classes )
	c=0
	for item in cm:
		resultsFile.write("%s " % classes[c])
		resultsFile.write("%s\n" % item)
		c+=1
	results.append(score)
	resultsFile.write(" avr accuracy %s\n" % float(results[i]))
	resultsFile.write(" accuracy %s\n" % float(acc))
	resultsFile.write(" precision %s\n" % float(prec))
	resultsFile.write(" recall %s\n" % float(recall))
	resultsFile.write(" featuers importance %s\n" % rfc.feature_importances_)
	importance.append(rfc.feature_importances_)
	print subjects[i], results[i]
	print rfc.feature_importances_

	#save models
	if not os.path.exists(projectDir+'/models'):
		os.makedirs(projectDir+'/models')
	joblib.dump(rfc,projectDir+'/models/'+subjects[i]+'.pk1')
#save score and avr in results file

Total=Tp+Tn+Fp+Fn
acc=100*(Tp+Tn)/Total
prec=100*Tp/(Tp+Fp)
recall=100*Tp/(Tp+Fn)
meanScore=mean(results)
resultsFile.write("=================================================================================\n")
resultsFile.write("=================================================================================\n")
resultsFile.write(" %s\n" % classes )
c=0

for itm in CM:
	resultsFile.write("%s " % classes[c])
	resultsFile.write("%s\n" % itm)
	c+=1

print "===================================CM=============================================="
print CM
print "===================================importanc=============================================="
print importance 
resultsFile.write("avr accuracy= %s\n" % float(meanScore ))
print "avr= " , meanScore 
resultsFile.write(" accuracy %s\n" % float(acc))
resultsFile.write(" precision %s\n" % float(prec))
resultsFile.write(" recall %s\n" % float(recall))
for itm in importance:

	resultsFile.write("%s\n" % itm)
	


	
	
