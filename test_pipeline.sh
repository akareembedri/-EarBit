#pipline:
fs=10
s=1
# frame generation(updat code->runUI->prepare)
	
n=60
a=10

# continue until $n equals 5
#while [ $a -le $n ]
#do
#echo "agg win $a"
while [ $fs -le $n ]
do
	echo "Start of $n sec frames."

#aware home
	#cd datasets/awer_home/
	#sh var_runUserDependent.sh $fs $s
	#sh var_prepareDir.sh $fs
	#cd ../..
	echo " $fs sec frames awarehome generated"
#wild
	#cd datasets/goPro_wild/
	#sh var_runUserDependent.sh $fs $s
	#sh var_prepareDir.sh $fs
	#cd ../..
	#echo " $fs sec frames wild generated"

#featuers generation (make code with and code without and run speratly)
#with mean Meadian
	#python var_statistical_data_gen.py $fs
	echo "stat featuers of $fs frames generated"
#without mean Media
	#python var_statistical_data_gen_woMeanMedian.py $fs

# frame level testing (with RF or SVM  classifiers)
#loo
	python var_RF_training_and_testing.py $fs
	echo "RF loo models of $fs frames generated"
#all for wild testing 
	#python var_RF_allModelGen.py $fs
	echo "RF All models of $fs frames generated"
#sequential data generation and testing
#awer home
	#python var_seqentialData_FeatGen_TestRF.py $fs
	echo "sequential stat data-awerhome- of $fs frames generated"
	#python var_seqentialData_FeatGen_TestRF_woMeanMedian.py $fs
#wild
	#python var_wild_seqDataGen.py $fs
	echo "sequential stat data-wild- of $fs frames generated"
# Event level testing 
# results generation and plotting 
	#python var_seqentialResults_Gen_vis.py $fs $a

#wild
	#python var_wild_visSeqDataGen.py $fs

	fs=$(( fs+10 ))	 # increments $fs
done
#fs=10
#a=$(( a+10 ))	 # increments $fs
#done
python var_RF_training_and_testing.py 5
