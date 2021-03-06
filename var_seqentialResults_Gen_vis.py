#This code loads seqential results and annotations
#allow results filtering
#plot binarry victors or confidance
#compute accuracy recall and precision
#!/usr/bin/python
import sys
import numpy as np
import os
from statistics import mean, variance, median,stdev
from math import sqrt
from sklearn.externals import joblib
import matplotlib.pyplot as plt

from collections import Counter

winSize=sys.argv[1]
agWin=sys.argv[2]

dataSetName=winSize+'sec_1sSlid_DS_lpf_all_4C_vR'
genDataSet='/home/mgoel1/Dropbox/EarBit/Statistical_Datasets/'+dataSetName+'_EarNeck7fEntropy/'
projectDir='/media/mgoel1/USB20FD/kareemsData/datasets/awer_home/'
subjects = ['April2Sub1','April2Sub3', 'April2Sub4', 'March24Sub1','March24Sub3','March24Sub4','March25Sub' , 'March28Sub1', 'March28Sub3', 'March28Sub4']
#subjects=['angela_1']#,'kareem_1/1','kareem_2/1','Kareem_2','Malcolm_1','Malcolm_2/30sec','richard/a','richard/b','sarah/a','sarah/b','lu/b','lu/a','lin/a','lin/b','Cheng_1/30sec','Cheng_2/30sec','jz_1','sharly_1']

types  =['testDataset']#[ 'eating' , 'silent' , 'talking' , 'walking']#'drinking' ['other','walking'] # 

mainClass='eating'

classes=['eating', 'silent', 'talking', 'walking']
annType=['eating','stationary','talking','moving','drinking','other']
CM=[[0 for col in range(len(classes))] for row in range(len(classes))]
Tp=0
Tn=0
Fp=0
Fn=0
results=[]
resultsFileName=genDataSet+agWin+'awerEventLevelResults.txt'
resultsFile=open(resultsFileName,'w')

#event var
AllEtp=0
AllEfn=0
AllEfp=0
AllOffset=[]
AllOnset=[]
AllCoverage=[]
AllOffsetPer=[]
AllEduration=[]
duration=[]
def mergEvents(data,win,value):
	startTime=0
	endTime=0
	eventEnds=False 
	eventStarts=False
	for f in range(0,len(data[0])):
		#print f,'----------------------------------------'
		for e in range(0,len(data)-1):
			#print e 
			if data[e][f]==value and data[e+1][f]==0 and eventEnds==False:
				#print e, ' event ended'
				eventEnds=True
				endTime=e+1
			if data[e][f]==0 and data[e+1][f]==value and eventEnds==True:
				#print e, ' event started'
				eventStarts=True
				startTime=e
			if eventStarts==True:
				eventStarts=False
				eventEnds=False 
				dif=startTime-endTime
				if dif<=win:
					#print dif,win,startTime,endTime	
					
					for m in range(endTime,startTime+1):
						#print data[m][f]
						data[m][f]=value
						#print data[m][f]
						#print '----------------------------------------------'
					
		startTime=0
		endTime=0
		eventEnds=False 
		eventStarts=False
	return data
def column(matrix, i):
    return [row[i] for row in matrix]

def movingaverage(interval, window_size):
    window = np.ones(int(window_size))/float(window_size)
    return np.convolve(interval, window, 'same')

def readConfig(path):
	f = open(path+"offsetConfig.txt", "rb")
	lines = f.readlines()
	lines = lines[1:]
	config = {}
	for line in lines:
		filetype, filename, offset = line.split(",")
		config[filetype] = (filename, offset)
	f.close()
	return config

def majortyVote(data,win):
	def popVote(dt):
		cdt=Counter(dt)
		return  cdt.most_common(1)[0][0]

	bound=int(win/2)
	#print bound

	for t in range(len(data[0])):
		#print t
		vec=column(data, t)
		for v in range(bound,len(vec)-bound):	
			data[v][t]=popVote(vec[v-bound:v+bound])
			#print returnData[v],popVote(vec[v-bound:v+bound]),t
	#print column(data, 3)
	return data

def movingAVR(data,win,threshold,value):
	bound=int(win/2)
	#print bound

	for t in range(len(data[0])):
		#print t
		vec=column(data, t)
		for v in range(bound,len(vec)-bound):	
			temp=mean(vec[v-bound:v+bound])	
			if temp >=threshold*value:
				data[v][t]=value
			else:
				data[v][t]=0
			#print returnData[v],popVote(vec[v-bound:v+bound]),t
	#print column(data, 3)
	return data
def eventLevelAnalysis(groundTruth,prediction):
	tp=0
	tn=0
	fp=0
	fn=0
	prediction=prediction[12:-12]
	groundTruth[0]=0
	groundTruth[-1]=0
	prediction[0]=0
	prediction[-1]=0
	# get the begining and end of the event index->if pred has 1 in that region its tp->look at the beging time and the end time for that event compute delay,tp and fn
	#compute fps the same way
	startTime=0
	endTime=0
	eventEnds=False 
	eventStarts=False
	offset=[]
	onset=[]
	offsetPer=[]
	c=0
	meanOffset=0
	meanOnset=0
	meanOffsetPer=0
	gtEvnDuration=[]
	predEvnDuration=[]
	FPpredEvnDuration=[]
	coverage=[]
	Eduration=[]
	#print groundTruth
	for e in range(0,len(groundTruth)-1):
			#print e 
		if groundTruth[e]==0 and groundTruth[e+1]>0 and eventStarts==False:
			#print e, ' event started'
			eventStarts=True
			startTime=e+1
		if groundTruth[e]>0 and groundTruth[e+1]==0 and eventStarts==True:
			#print e, ' event ended'
			eventStarts=False
			eventEnds=True
			endTime=e
			
		if eventEnds==True:
			eventEnds=False 
			Eduration.append(endTime-startTime)
			tempID=[i for i, z in enumerate(prediction[startTime:endTime]) if z >0]
			if len(tempID)>0:
				tp+=1
				offset.append((startTime+tempID[0])-startTime)
				#onset.append( endTime-(endTime-tempID[-1]))
				onset.append(len(prediction[startTime:endTime])-tempID[-1])
				predEvnDuration.append(len(tempID))
				offsetPer.append(float(float((startTime+tempID[0])-startTime)/float(endTime-startTime)))
			else:
				fn=+1
				predEvnDuration.append(0)
			
			gtEvnDuration.append(endTime-startTime)
			coverage.append(float(float(len(tempID))/float(endTime-startTime)))
			c+=1


	print "number of events ",c		 
	print "captured ", tp
	print "missed ", fn
	print "gt durations ", gtEvnDuration
	print "predicted durations ", predEvnDuration
	if len(offset)>0:
		meanOffset=mean(offset)
		meanOnset=mean(onset) 
		meanOffsetPer=mean(offsetPer)
	for t in range(0,len(prediction)-1):
			#print e 
		if prediction[t]==0 and prediction[t+1]>0 and eventStarts==False:
			#print e, ' event started'
			eventStarts=True
			startTime=t+1
		if prediction[t]>0 and prediction[t+1]==0 and eventStarts==True:
			#print e, ' event ended'
			eventStarts=False
			eventEnds=True
			endTime=t
			
		if eventEnds==True:
			eventEnds=False 
			c+=1
			tempID=[i for i, z in enumerate(groundTruth[startTime:endTime]) if z >0]
			if len(tempID)==0:
				fp=+1
				FPpredEvnDuration.append(endTime-startTime)
				print "Fp st en: ",startTime,endTime
	print 'FP events ',fp,FPpredEvnDuration
	fp=len(FPpredEvnDuration)
	#print "mean offset ", meanOffset,"meanOnset ",meanOnset 
	return Eduration,offset,meanOffset,offsetPer,meanOffsetPer,onset,meanOnset,tp,fp,fn,gtEvnDuration,predEvnDuration,FPpredEvnDuration,coverage

#loads seqential data frame by frame-generate featuers and save them in text file
we=0;
for i in range(len(subjects)):
	#load predictions
	print subjects[i]
	predictFile=open(genDataSet+subjects[i]+'_predictResults.txt','r')
	pfile=map(str.split,predictFile)
	pfile=pfile[1:]
	predStr=column(pfile, 0)
	#print pfile
	for k in range(len(predStr)):
		#print predStr[i]
		pfile[k][0]=str(classes.index(pfile[k][0]))
	
	pfile=[[float(y) for y in x] for x in pfile]
	#generating prediction vectors
	predVec=[]
	#ov=[[0]*(len(classes))]
	for e in range(len(pfile)):
		predVec+=[[0]*(len(classes))]
	#print predVec
	for j in range(len(pfile)):	
		#print 	predVec[j][int(pfile[j][0])]
		predVec[j][int(pfile[j][0])]=0.75
		
	#print predVec
	#load annotations
	
	config = readConfig(projectDir+subjects[i]+'/ALL/')
	file_annotations, offset_annotations = config["annotations"]
	file_earbitData, offset_earbitData = config["earbitData"]
	f_annotations = open(projectDir+subjects[i]+'/ALL/'+file_annotations,'r')
	lines=f_annotations.readlines()
	annfile=[]
	for line in lines:
		annfile.append(line.strip().split(","))
	annClass=annfile[0]
	#flip the talking and drinking annoation back
	if annClass==['time','moving','stationary','eating','drinking','talking','other']:
		#print annfile[930:940]
		for a in range(len(annfile)):
			temp=annfile[a][4]
			annfile[a][4]=annfile[a][5]
			annfile[a][5]=temp
		#print annfile[930:940]
	annfile=annfile[1:]
	annfile=[[float(y) for y in x] for x in annfile]
	#add offset
	#print len(annfile[0]),len(predVec[0]),len(pfile[0])
	#print len(annfile),len(predVec),len(pfile)
	offset=int(float(offset_earbitData.split('\n')[0]))
	print offset,offset_earbitData
	if i==0 :
		offset-=200
	if i==1 :
		offset-=320
	if offset>=0:
		#annfile=[[0,0,1,0,0,0,0]]*abs(offset)+annfile
		annfile=annfile[offset-1:]
	else:
		predVec=[[0,0.75,0,0]]*abs(offset)+predVec
		pfile=[[0,0,0,0,0]]*abs(offset)+pfile
	#print len(annfile),len(predVec),len(pfile)
#---------------------------------------------------------------------------------------------------------------
	# filter
	'''
	threshold=0.15
	win=60*5#sec
	#annfile=majortyVote(annfile,win)#movingAVR(annfile,win,threshold)
	#annfile=movingAVR(annfile,win,threshold,1)
	annfile=mergEvents(annfile,win,1)
	predVec=majortyVote(predVec,30)#movingAVR(annfile,win,threshold)
	#predVecTemp=predVec
	
	predVec=movingAVR(predVec,30,threshold,0.75)
	predVec=mergEvents(predVec,60*3,0.75)
	#pfile=movingAVR(pfile,30,0.75,0.5)
	#print column(predVec, 3)
	#pfile=Majortyfilter(
	'''
	################################
	threshold=0.1
	win=60*10#sec
	#annfile=majortyVote(annfile,win)#movingAVR(annfile,win,threshold)
	#annfile=movingAVR(annfile,win,threshold,1)
	annfile=mergEvents(annfile,win,1)
	#predVec=majortyVote(predVec,30)#movingAVR(annfile,win,threshold)
	#predVecTemp=predVec
	
	#predVec=movingAVR(predVec,30,threshold,0.75)
	#predVec=mergEvents(predVec,60*3,0.75)
	
	pfile=majortyVote(pfile,int(agWin))
	pfile=movingAVR(pfile,25,0.85,0.65)
	pfile=majortyVote(pfile,int(agWin))
	pfile=mergEvents(pfile,win,0.65)
	
#---------------------------------------------------------------------------------------------------------------
	# plot
	'''
	for cls in [mainClass]:#classes:
		ci=classes.index(cls)
		plt.figure(ci)
		#print ci
		#annotation 
		plt.plot(range(len(annfile)),column(annfile,annClass.index(annType[ci])))
		#prediction
		plt.plot(range(len(predVec)),column(predVec, ci))
		#confidance 
		#plt.plot(range(len(pfile)),column(pfile, ci+1))
		#filtered		
		#plt.plot(range(len(fpredVec)),column(fpredVec,ci))

		plt.legend(['annotation', 'prediction','confidence'], loc='upper right')
		axes = plt.gca()
		#axes.set_xlim([xmin,xmax])
		axes.set_ylim([-0.1,1.2])
		plt.show()	
		
	for cls in [mainClass]:#classes:
		ci=classes.index(cls)
		plt.figure(ci)
		#print ci
		#annotation 
		plt.plot(range(len(annfile)),column(annfile,annClass.index(annType[ci])))
		#prediction
		#plt.plot(range(len(predVec)),column(predVec, ci))
		#confidance 
		plt.plot(range(len(pfile)),column(pfile, ci+1))
		#filtered		
		#plt.plot(range(len(fpredVec)),column(fpredVec,ci))

		plt.legend(['annotation', 'prediction','confidence'], loc='upper right')
		axes = plt.gca()
		#axes.set_xlim([xmin,xmax])
		axes.set_ylim([-0.1,1.2])
		plt.show()
		'''	
#---------------------------------------------------------------------------------------------------------------
	# comopute results
	tp=0
	tn=0
	fp=0
	fn=0	
	cm=[[0 for col in range(len(classes))] for row in range(len(classes))]

	for cls in ['eating']:
		ci=classes.index(cls)
		if len(annfile)<=len(pfile):
			length=len(annfile)
		else:
			length=len(pfile)
		duration.append(length)
		for l in range(length):
			if (annfile[l][1]==1 and annfile[l][2]==0 and annfile[l][3]==1):
				we+=1
			if (annfile[l][1]+annfile[l][2]+annfile[l][3]+annfile[l][4]+annfile[l][5]+annfile[l][6])!=0: # if we know the lable
				testLabels=annfile[l][annClass.index(annType[ci])]
				resultsValues=pfile[l][classes.index(cls)+1]
				if testLabels==1 and resultsValues>0.1 :
					tp+=1
				if testLabels==1 and resultsValues==0 :
					fn+=1
				if testLabels==0 and resultsValues==0:
					tn+=1
				if testLabels==0 and resultsValues>0.1 :
					fp+=1
				#print annfile[l]
				if 1 in annfile[l] and 0.65 in pfile[l]:
					tempID=[i for i, e in enumerate(annfile[l]) if e == 1][0]
					#print annfile[l], tempID
					if tempID !=0 and (annfile[l][5]!=1 or annfile[l][6]!=1): # avoid time stampe with value 1 other and drinking

						if tempID==2 and annfile[l][3]==1: # if stationarry check if there is eating or drinking 
							tempID=3
						elif tempID==2 and annfile[l][4]==1:
							tempID=4
						testL=annType.index(annClass[tempID])
						predL=[i for i, e in enumerate(pfile[l]) if e != 0][0]
						if testL<4:
							cm[testL][predL]+=1
		
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
	resultsFile.write("=================================================================================\n")			
	resultsFile.write(" %s\n" % subjects[i] )
	resultsFile.write(" %s\n" % classes )
	c=0
	for item in cm:
		resultsFile.write("%s " % classes[c])
		resultsFile.write("%s\n" % item)
		c+=1


	resultsFile.write(" accuracy %s\n" % float(acc))
	resultsFile.write(" precision %s\n" % float(prec))
	resultsFile.write(" recall %s\n" % float(recall))
	

	
	Eduration,Offset,MeanOffset,offsetPer,meanOffsetPer,Onset,MeanOnset,Etp,Efp,Efn,GTEvnDuration,PredEvnDuration,FPpredEvnDuration,Coverage=eventLevelAnalysis(column(annfile, annClass.index(annType[classes.index(mainClass)])),column(pfile, classes.index(mainClass)+1))	
	resultsFile.write(" ++ Event Level ++ \n")
	
	if (Etp+Efp)==0:
		prec=0
	else:
		prec=100*Etp/(Etp+Efp)
	if (Etp+Efn)==0:
		recall=0
	else:	
		recall=100*Etp/(Etp+Efn)
	
	#resultsFile.write(" accuracy %s\n" % Etp )
	resultsFile.write(" captured %s " % Etp)
	resultsFile.write("  of %s events\n" %(Etp+Efn))
	resultsFile.write(" falsely recognized %s events " % Efp)
	resultsFile.write(" durations %s\n" % FPpredEvnDuration)
	resultsFile.write(" precision %s\n" % float(prec))
	resultsFile.write(" recall/accuracy %s\n" % float(recall))
	resultsFile.write(" Offsets:  %s" % Offset)
	resultsFile.write("   Mean Offsets=  %s\n" % MeanOffset)
	resultsFile.write(" Offsets percentage:  %s" % offsetPer)
	resultsFile.write("   Mean Offsets percentage=  %s\n" % meanOffsetPer)
	resultsFile.write(" Onsets:  %s" % Onset)
	resultsFile.write("   Mean Onsets=  %s\n" % MeanOnset)
	resultsFile.write(" Coverage:  %s" % Coverage)
	resultsFile.write("   Mean Coverage=  %s\n" % mean(Coverage))
	AllEtp+=Etp
	AllEfn+=Efn
	AllEfp+=Efp
	AllOffset=AllOffset+Offset
	AllOnset=AllOnset+Onset
	AllCoverage=AllCoverage+Coverage
	AllOffsetPer=AllOffsetPer+offsetPer
	AllEduration=AllEduration+Eduration
Total=Tp+Tn+Fp+Fn
acc=100*(Tp+Tn)/Total
prec=100*Tp/(Tp+Fp)
recall=100*Tp/(Tp+Fn)

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

print "durations ", duration
print "total ", sum(duration)
print "eating+walking ",we
print 1.0*(we)/sum(duration)

print 'avr eating duration', mean(AllEduration)

resultsFile.write(" accuracy %s\n" % float(acc))
resultsFile.write(" precision %s\n" % float(prec))
resultsFile.write(" recall %s\n" % float(recall))

resultsFile.write(" ++ Event Level ++ \n")

resultsFile.write(" captured %s " % AllEtp)
resultsFile.write("  of %s events\n" %(AllEtp+AllEfn))
resultsFile.write(" falsely recognized %s events\n" % AllEfp)
if (AllEtp+AllEfp)==0:
	prec=0
else:
	prec=100*AllEtp/(AllEtp+AllEfp)
if (AllEtp+AllEfn)==0:
	recall=0
else:	
	recall=100*AllEtp/(AllEtp+AllEfn)
resultsFile.write(" precision %s\n" % float(prec))
resultsFile.write(" recall/accuracy %s\n" % float(recall))


	
resultsFile.write("   Mean Offsets=  %s" % mean(AllOffset))
resultsFile.write("   +/- %s\n" % stdev(AllOffset))
	
resultsFile.write("   Mean Offsets percentage =  %s" % mean(AllOffsetPer))
resultsFile.write("   +/- %s\n" % stdev(AllOffsetPer))

resultsFile.write("   Mean Onsets=  %s" % mean(AllOnset))
resultsFile.write("   +/- %s\n" % stdev(AllOnset))
	
resultsFile.write("   Mean Coverage=  %s" % mean(AllCoverage))
resultsFile.write("   +/- %s\n" % stdev(AllCoverage))
	
	

	
